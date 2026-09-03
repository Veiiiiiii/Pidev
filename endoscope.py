#!/usr/bin/env python3
"""
Endoscope viewer with a 3D pointing indicator
---------------------------------------------
Fullscreen video from the AtomS3R-CAM probe with a compass-style indicator
showing where the lens is aimed, for use where the probe is out of sight.

    python3 endoscope.py
    python3 endoscope.py --windowed        # for development on a desktop
    python3 endoscope.py --port /dev/ttyACM0

LAYOUT
    top-left      EXIT
    top-right     3D orientation indicator
    bottom-right  ZERO

HOW THE INDICATOR WORKS
    The probe reports a quaternion, not Euler angles: an endoscope routinely
    points straight down a hole, which is exactly where pitch/roll/yaw gimbal
    lock and the indicator would flip.

    Zeroing captures the current attitude as the reference. Everything drawn
    afterwards is the probe's direction *relative to that reference*, so the
    display never depends on which way the room faces -- only on how far the
    probe has turned since you zeroed it.

    Screen-up on the indicator is real-world up, taken from gravity, so the
    picture stays meaningful even though heading itself has no absolute source.

WHY THERE IS A ZERO BUTTON AND NOT A MAGNETOMETER
    The BMM150 could give absolute heading, but this probe works around steel:
    engine bays, pipework, machinery. Magnetic heading there is worse than
    gyro drift. Re-zeroing on demand is the honest fix, so the button stays
    reachable during use rather than only at startup.

SETUP ON THE PI
    sudo apt install python3-serial python3-opencv python3-pil.imagetk -y
    sudo usermod -aG dialout $USER      # then REBOOT
"""
import argparse
import glob
import json
import math
import os
import sys
import threading
import time

try:
    import serial
except ImportError:
    print("pyserial missing. Run: sudo apt install python3-serial -y",
          file=sys.stderr)
    sys.exit(1)

import numpy as np

try:
    import cv2
except ImportError:
    print("OpenCV missing. Run: sudo apt install python3-opencv -y",
          file=sys.stderr)
    sys.exit(1)

import tkinter as tk
import tkinter.font as tkfont
from PIL import Image, ImageTk


CONFIG = os.path.join(os.path.expanduser("~"), ".config", "endoscope.json")
SYNC = b"\xa5\x5a"
TYPE_IMU = 1
TYPE_FRAME = 2

# Which body axis the lens looks along. The probe can be mounted any way
# round, so this is chosen once during setup and remembered.
AXES = [
    ("+X", (1.0, 0.0, 0.0)),
    ("-X", (-1.0, 0.0, 0.0)),
    ("+Y", (0.0, 1.0, 0.0)),
    ("-Y", (0.0, -1.0, 0.0)),
    ("+Z", (0.0, 0.0, 1.0)),
    ("-Z", (0.0, 0.0, -1.0)),
]

TEXT = {
    "exit": "EXIT", "zero": "ZERO", "axis": "AXIS",
    "start": "ZERO AND START",
    "title": "Zero before use",
    "hint1": "Hold the probe level, lens facing the screen",
    "hint2": "Keep it still, then press the button below",
    "hint3": "If the preview below is not facing you, press AXIS first",
    "notzero": "NOT ZEROED", "still": "re-calibrating",
    "az": "AZ", "el": "EL", "screen": "SCREEN",
}


# ------------------------------------------------------------- quaternions

def q_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def q_conj(q):
    return (q[0], -q[1], -q[2], -q[3])


def q_rotate(q, v):
    """Rotate vector v by quaternion q."""
    w, x, y, z = q
    vx, vy, vz = v
    # t = 2 * (q_vec x v);  result = v + w*t + q_vec x t
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def v_norm(v):
    n = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if n < 1e-9:
        return (0.0, 0.0, 1.0)
    return (v[0] / n, v[1] / n, v[2] / n)


def v_cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def v_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


# ------------------------------------------------------------- serial link

def find_port():
    for pattern in ("/dev/ttyACM*", "/dev/ttyUSB*"):
        ports = sorted(glob.glob(pattern))
        if ports:
            return ports[0]
    return None


class ProbeLink:
    """
    Demultiplexes the probe's packet stream on a background thread and keeps
    only the newest frame and the newest attitude. Nothing queues up: for a
    live view, a late frame is worthless, so dropping is the correct policy.
    """

    def __init__(self, port=None, baud=115200):
        self.port = port or find_port()
        self.baud = baud
        self.ser = None
        self.buf = bytearray()
        self.lock = threading.Lock()
        self.frame = None            # BGR ndarray
        self.frame_seq = 0
        self.quat = (1.0, 0.0, 0.0, 0.0)
        self.still = False
        self.imu_seq = 0
        self.status = []
        self.bad_packets = 0
        self.running = False
        self.error = None
        self._fps_times = []

    def start(self):
        if self.port is None:
            raise RuntimeError("No probe found. Check the USB-C cable carries "
                               "data, then run: ls /dev/ttyACM*")
        self.ser = serial.Serial(self.port, self.baud, timeout=0.5)
        time.sleep(2.0)              # the board reboots when the port opens
        self.ser.reset_input_buffer()
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()
        return self

    def stop(self):
        self.running = False
        time.sleep(0.1)
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass

    def video_fps(self):
        now = time.time()
        self._fps_times = [t for t in self._fps_times if now - t < 2.0]
        return len(self._fps_times) / 2.0

    def _next_packet(self):
        while True:
            idx = self.buf.find(SYNC)
            if idx < 0:
                if len(self.buf) > 1:
                    del self.buf[:-1]        # sync word may straddle reads
                return None
            if idx > 0:
                del self.buf[:idx]
            if len(self.buf) < 7:
                return None

            ptype = self.buf[2]
            length = int.from_bytes(self.buf[3:7], "little")
            if length > 512 * 1024:          # implausible -> resynced wrongly
                del self.buf[:2]
                self.bad_packets += 1
                continue
            total = 7 + length + 1
            if len(self.buf) < total:
                return None

            payload = bytes(self.buf[7:7 + length])
            checksum = self.buf[7 + length]
            del self.buf[:total]

            calc = 0
            for b in payload:
                calc ^= b
            if calc != checksum:
                self.bad_packets += 1
                continue
            return ptype, payload

    def _loop(self):
        while self.running:
            try:
                chunk = self.ser.read(self.ser.in_waiting or 1)
            except Exception as e:
                self.error = str(e)
                self.running = False
                return
            if chunk:
                self.buf.extend(chunk)

            while True:
                pkt = self._next_packet()
                if pkt is None:
                    break
                ptype, payload = pkt

                if ptype == TYPE_IMU:
                    try:
                        data = json.loads(payload.decode("utf-8", "ignore"))
                    except json.JSONDecodeError:
                        continue
                    if "status" in data:
                        self.status.append(data)
                        continue
                    q = data.get("q")
                    if q and len(q) == 4:
                        with self.lock:
                            self.quat = (q[0], q[1], q[2], q[3])
                            self.still = bool(data.get("st"))
                            self.imu_seq += 1

                elif ptype == TYPE_FRAME:
                    arr = np.frombuffer(payload, dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        with self.lock:
                            self.frame = img
                            self.frame_seq += 1
                        self._fps_times.append(time.time())

    def snapshot(self):
        with self.lock:
            return self.frame, self.frame_seq, self.quat, self.still


# ------------------------------------------------------------- indicator

class Indicator:
    """
    Draws the probe as a slim body with a black lens block and a pale arrow
    ahead of it, oriented in 3D from a fixed centre like a compass needle.

    Perspective is deliberate rather than decorative: when the probe aims at
    you the shape foreshortens to almost nothing, and when it aims away it
    stretches out. That difference is the only cue that separates "towards"
    from "away", which a flat 2D compass cannot show at all.
    """

    def __init__(self, canvas, cx, cy, radius, strings):
        self.c = canvas
        self.cx, self.cy, self.R = cx, cy, radius
        self.s = strings
        self.depth = 3.2                      # eye distance, in radius units
        self.items = []

    def _proj(self, p):
        f = self.depth / max(self.depth - p[2], 0.25)
        return (self.cx + p[0] * self.R * f,
                self.cy - p[1] * self.R * f)

    def clear(self):
        for i in self.items:
            self.c.delete(i)
        self.items = []

    def _poly(self, pts3, **kw):
        flat = []
        for p in pts3:
            x, y = self._proj(p)
            flat += [x, y]
        self.items.append(self.c.create_polygon(*flat, **kw))

    def draw(self, direction, up_disp, zeroed):
        """direction and up_disp are unit vectors in display coordinates."""
        self.clear()
        c, R = self.c, self.R

        # Outer boundary and centre.
        self.items.append(c.create_oval(self.cx - R, self.cy - R,
                                        self.cx + R, self.cy + R,
                                        outline="#3c4a52", width=2))

        # Ring lying in the real-world horizontal plane: the tilt of this
        # ellipse tells you how the probe is inclined relative to level.
        h1 = v_cross(up_disp, (0.0, 0.0, 1.0))
        if math.sqrt(v_dot(h1, h1)) < 1e-3:
            h1 = v_cross(up_disp, (1.0, 0.0, 0.0))
        h1 = v_norm(h1)
        h2 = v_norm(v_cross(up_disp, h1))
        ring = []
        for k in range(37):
            a = 2 * math.pi * k / 36
            w = (0.82 * (math.cos(a) * h1[0] + math.sin(a) * h2[0]),
                 0.82 * (math.cos(a) * h1[1] + math.sin(a) * h2[1]),
                 0.82 * (math.cos(a) * h1[2] + math.sin(a) * h2[2]))
            ring += list(self._proj(w))
        self.items.append(c.create_line(*ring, fill="#2f6f7a", width=1,
                                        smooth=True))

        if not zeroed:
            self.items.append(c.create_text(
                self.cx, self.cy, text=self.s["notzero"],
                fill="#ff8a65", font=("DejaVu Sans", max(9, R // 7))))
            return

        d = v_norm(direction)

        # A ribbon that always faces the viewer stands in for a solid body:
        # cheap to draw, and at these sizes indistinguishable from a box.
        side = v_cross(d, (0.0, 0.0, 1.0))
        if math.sqrt(v_dot(side, side)) < 1e-3:
            side = v_cross(d, (0.0, 1.0, 0.0))
        side = v_norm(side)

        def at(t, off=0.0):
            return (d[0] * t + side[0] * off,
                    d[1] * t + side[1] * off,
                    d[2] * t + side[2] * off)

        away = d[2] < 0                      # pointing behind the screen
        body_fill = "#4a5f6b" if away else "#7ea8bd"
        body_line = "#2b3b44" if away else "#cfe4ef"

        w = 0.085
        self._poly([at(0.02, w), at(0.62, w), at(0.62, -w), at(0.02, -w)],
                   fill=body_fill, outline=body_line, width=1)

        # Black block = the lens end. Whichever end this is on is the end
        # you are looking out of.
        bw = 0.115
        self._poly([at(0.62, bw), at(0.80, bw), at(0.80, -bw), at(0.62, -bw)],
                   fill="#0d0d0d", outline="#000000", width=1)

        # Pale arrow beyond the lens, drawn as nested slices so it reads as a
        # gradient without needing an image layer.
        tip = at(1.16)
        grays = ["#7d8a91", "#98a5ac", "#b4c1c8", "#d2dee4", "#f2f7fa"]
        n = len(grays)
        for i, g in enumerate(grays):
            t0 = 0.90 + (1.16 - 0.90) * (i / n)
            aw = 0.20 * (1.0 - i / n)
            self._poly([at(t0, aw), at(t0, -aw), tip], fill=g, outline="")

        # Head-on marker: with no length left to see, the ring alone would be
        # ambiguous, so mark the centre explicitly.
        if abs(d[2]) > 0.985:
            r = R * 0.10
            self.items.append(c.create_oval(
                self.cx - r, self.cy - r, self.cx + r, self.cy + r,
                fill="#f2f7fa" if d[2] > 0 else "#2b3b44",
                outline="#0d0d0d", width=2))


# ------------------------------------------------------------- application

class App:
    def __init__(self, link, args):
        self.link = link
        self.args = args
        self.cfg = self.load_cfg()
        self.axis_idx = self.cfg.get("axis", 0)
        self.q_ref = None
        self.last_seq = -1
        self.running = True

        self.root = tk.Tk()
        self.root.title("Endoscope")
        self.root.configure(bg="black")
        if not args.windowed:
            self.root.attributes("-fullscreen", True)
            self.root.configure(cursor="none")
        else:
            self.root.geometry("1024x600")
        self.root.update_idletasks()

        self.W = self.root.winfo_width() if args.windowed else self.root.winfo_screenwidth()
        self.H = self.root.winfo_height() if args.windowed else self.root.winfo_screenheight()

        self.pick_font()
        self.s = TEXT

        self.canvas = tk.Canvas(self.root, width=self.W, height=self.H,
                                bg="black", highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)

        self.video_item = self.canvas.create_image(self.W // 2, self.H // 2)
        self.photo = None

        self.build_hud()
        self.build_zero_screen()

        self.root.bind("<Escape>", lambda e: self.quit())
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

    # ---- persistence

    def load_cfg(self):
        try:
            with open(CONFIG, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def save_cfg(self):
        try:
            os.makedirs(os.path.dirname(CONFIG), exist_ok=True)
            with open(CONFIG, "w", encoding="utf-8") as f:
                json.dump({"axis": self.axis_idx}, f)
        except Exception:
            pass

    def pick_font(self):
        """Pick a clean sans face; the UI text is English either way."""
        preferred = ("DejaVu Sans", "Liberation Sans", "Noto Sans", "Arial")
        have = set(tkfont.families())
        for name in preferred:
            for fam in have:
                if name.lower() == fam.lower():
                    self.font_family = fam
                    return
        self.font_family = "TkDefaultFont"

    def f(self, size, bold=False):
        return (self.font_family, size, "bold") if bold else (self.font_family, size)

    # ---- HUD

    def button(self, x, y, w, h, label, fill, cb, size=None):
        size = size or max(11, h // 3)
        r = self.canvas.create_rectangle(x, y, x + w, y + h, fill=fill,
                                         outline="", width=0)
        t = self.canvas.create_text(x + w / 2, y + h / 2, text=label,
                                    fill="white", font=self.f(size, True))
        for item in (r, t):
            self.canvas.tag_bind(item, "<Button-1>", lambda e: cb())
        return r, t

    def build_hud(self):
        pad = max(8, self.H // 50)
        bw = max(90, self.W // 9)
        bh = max(46, self.H // 11)

        self.hud = []
        self.hud += list(self.button(pad, pad, bw, bh, "✕ " + self.s["exit"],
                                     "#c62828", self.quit))
        self.hud += list(self.button(self.W - bw - pad, self.H - bh - pad,
                                     bw, bh, self.s["zero"], "#1565c0",
                                     self.zero_now))

        panel = int(min(self.W, self.H) * 0.34)
        px = self.W - panel - pad
        py = pad
        self.hud.append(self.canvas.create_rectangle(
            px, py, px + panel, py + panel, fill="#0b1114", outline="#37474f",
            width=2))

        self.indicator = Indicator(self.canvas, px + panel / 2,
                                   py + panel / 2 - panel * 0.04,
                                   panel * 0.36, self.s)
        self.readout = self.canvas.create_text(
            px + panel / 2, py + panel - max(12, panel // 12),
            text="", fill="#90a4ae", font=self.f(max(9, panel // 18)))
        self.hud.append(self.readout)

        self.statusbar = self.canvas.create_text(
            pad, self.H - pad, anchor="sw", text="", fill="#607d8b",
            font=self.f(max(9, self.H // 60)))
        self.hud.append(self.statusbar)

    # ---- zeroing screen

    def build_zero_screen(self):
        self.zero_items = []
        z = self.zero_items
        c = self.canvas

        z.append(c.create_rectangle(0, 0, self.W, self.H, fill="#06090b",
                                    outline=""))
        z.append(c.create_text(self.W / 2, self.H * 0.10, text=self.s["title"],
                               fill="#ffffff", font=self.f(max(16, self.H // 18), True)))
        z.append(c.create_text(self.W / 2, self.H * 0.19, text=self.s["hint1"],
                               fill="#b0bec5", font=self.f(max(11, self.H // 30))))
        z.append(c.create_text(self.W / 2, self.H * 0.25, text=self.s["hint2"],
                               fill="#b0bec5", font=self.f(max(11, self.H // 30))))

        # Schematic: probe on the left aimed at a screen on the right.
        cy = self.H * 0.40
        x0 = self.W * 0.30
        x1 = self.W * 0.66
        z.append(c.create_rectangle(x1, cy - self.H * 0.07,
                                    x1 + self.W * 0.07, cy + self.H * 0.07,
                                    outline="#546e7a", width=3))
        z.append(c.create_text(x1 + self.W * 0.035, cy + self.H * 0.11,
                               text=self.s["screen"], fill="#546e7a",
                               font=self.f(max(9, self.H // 45))))
        z.append(c.create_rectangle(x0 - self.W * 0.09, cy - self.H * 0.018,
                                    x0, cy + self.H * 0.018,
                                    fill="#7ea8bd", outline=""))
        z.append(c.create_rectangle(x0, cy - self.H * 0.025,
                                    x0 + self.W * 0.02, cy + self.H * 0.025,
                                    fill="#0d0d0d", outline=""))
        z.append(c.create_line(x0 + self.W * 0.03, cy, x1 - self.W * 0.02, cy,
                               fill="#cfd8dc", width=3, arrow="last",
                               arrowshape=(16, 20, 7)))

        # Live preview so the axis can be confirmed before committing.
        prev = int(min(self.W, self.H) * 0.30)
        pxc = self.W / 2
        pyc = self.H * 0.63
        z.append(c.create_oval(pxc - prev / 2, pyc - prev / 2,
                               pxc + prev / 2, pyc + prev / 2,
                               outline="#263238", width=1))
        self.preview = Indicator(c, pxc, pyc, prev * 0.36, self.s)

        z.append(c.create_text(self.W / 2, self.H * 0.80, text=self.s["hint3"],
                               fill="#78909c", font=self.f(max(9, self.H // 40))))

        bw = max(110, self.W // 7)
        bh = max(48, self.H // 11)
        z += list(self.button(self.W / 2 - bw - 12, self.H * 0.87, bw, bh,
                              self.s["axis"] + "  " + AXES[self.axis_idx][0],
                              "#455a64", self.cycle_axis))
        self.axis_btn_text = z[-1]
        z += list(self.button(self.W / 2 + 12, self.H * 0.87, bw, bh,
                              self.s["start"], "#2e7d32", self.finish_zero))

        self.zeroing = True

    def cycle_axis(self):
        self.axis_idx = (self.axis_idx + 1) % len(AXES)
        self.save_cfg()
        self.canvas.itemconfigure(
            self.axis_btn_text,
            text=self.s["axis"] + "  " + AXES[self.axis_idx][0])

    def finish_zero(self):
        _, _, q, _ = self.link.snapshot()
        self.q_ref = q
        self.zeroing = False
        self.preview.clear()
        for i in self.zero_items:
            self.canvas.delete(i)
        self.zero_items = []

    def zero_now(self):
        _, _, q, _ = self.link.snapshot()
        self.q_ref = q

    # ---- geometry

    def display_vectors(self, q_now):
        """
        Express the probe's current aim, and real-world up, in a frame where
        the zeroed aim points straight out of the screen.
        """
        fwd = AXES[self.axis_idx][1]
        v_now = v_norm(q_rotate(q_now, fwd))

        if self.q_ref is None:
            return v_now, (0.0, 1.0, 0.0)

        e3 = v_norm(q_rotate(self.q_ref, fwd))
        world_up = (0.0, 0.0, 1.0)
        e1 = v_cross(world_up, e3)
        if math.sqrt(v_dot(e1, e1)) < 1e-3:      # aimed at the zenith/nadir
            e1 = v_cross((1.0, 0.0, 0.0), e3)
        e1 = v_norm(e1)
        e2 = v_norm(v_cross(e3, e1))

        def to_disp(v):
            return (v_dot(v, e1), v_dot(v, e2), v_dot(v, e3))

        return to_disp(v_now), v_norm(to_disp(world_up))

    # ---- main loop

    def update(self):
        if not self.running:
            return

        frame, seq, q, still = self.link.snapshot()

        if frame is not None and seq != self.last_seq and not self.zeroing:
            self.last_seq = seq
            h, w = frame.shape[:2]
            scale = min(self.W / w, self.H / h)
            resized = cv2.resize(frame, (int(w * scale), int(h * scale)),
                                 interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.canvas.itemconfigure(self.video_item, image=self.photo)
            self.canvas.tag_lower(self.video_item)

        d, up = self.display_vectors(q)

        if self.zeroing:
            # Preview treats "now" as the reference so the operator can see
            # whether the chosen axis really points at them.
            fwd = AXES[self.axis_idx][1]
            v = v_norm(q_rotate(q, fwd))
            self.preview.draw(v if self.q_ref else self._preview_dir(v),
                              (0.0, 1.0, 0.0), True)
        else:
            self.indicator.draw(d, up, self.q_ref is not None)
            az = math.degrees(math.atan2(d[0], max(d[2], -1.0)))
            el = math.degrees(math.asin(max(-1.0, min(1.0, d[1]))))
            self.canvas.itemconfigure(
                self.readout,
                text="{} {:+.0f}°   {} {:+.0f}°".format(
                    self.s["az"], az, self.s["el"], el))
            self.canvas.itemconfigure(
                self.statusbar,
                text="{:.0f} fps    {}".format(
                    self.link.video_fps(),
                    self.s["still"] if still else ""))

        self.root.after(40, self.update)

    def _preview_dir(self, v):
        """Before zeroing there is no reference, so show the raw body axis."""
        return v

    def quit(self):
        self.running = False
        try:
            self.root.destroy()
        except Exception:
            pass

    def run(self):
        self.root.after(100, self.update)
        self.root.mainloop()


def main():
    ap = argparse.ArgumentParser(description="Endoscope viewer with 3D aim indicator")
    ap.add_argument("--port", help="e.g. /dev/ttyACM0 (auto-detected if omitted)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--windowed", action="store_true",
                    help="run in a window instead of fullscreen")
    args = ap.parse_args()

    try:
        link = ProbeLink(args.port, args.baud).start()
    except Exception as e:
        print(e, file=sys.stderr)
        return 1

    print(f"Probe on {link.port}")
    app = App(link, args)
    try:
        app.run()
    finally:
        link.stop()
        if link.status:
            print("\nProbe reported:")
            for s in link.status:
                print(" ", s)
        if link.bad_packets:
            print(f"({link.bad_packets} corrupted packets discarded)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
