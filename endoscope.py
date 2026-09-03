#!/usr/bin/env python3
"""
Endoscope viewer with a 3D aim indicator
----------------------------------------
Fullscreen video from the AtomS3R-CAM probe plus a compass showing where the
lens is aimed, for work where the probe is out of sight.

    python3 endoscope.py
    python3 endoscope.py --windowed        # for development on a desktop
    python3 endoscope.py --port /dev/ttyACM0

LAYOUT
    top-left      EXIT
    top-right     aim indicator
    bottom-right  ZERO

HOW THE INDICATOR READS
    An azimuthal projection of the whole sphere onto a disc. The rim is
    horizontal, the centre is vertical, and the top of the disc is the
    direction you were facing when you zeroed.

        distance from centre   how close to horizontal the lens is
        angle around the disc  how far left or right of your zero heading
        arrow size             elevation: it grows aiming up, shrinks aiming down
        solid dot at centre    straight up
        hollow ring at centre  straight down

    Length alone cannot separate "aimed up" from "aimed down" -- both
    foreshorten identically -- so the arrow scale carries that sign. Reading
    the disc as though you were looking down on the probe from above, aiming
    up brings the tip toward you and it draws larger.

ELEVATION IS ABSOLUTE, HEADING IS RELATIVE
    Elevation comes from gravity, so it never drifts and never needs zeroing.
    Only the heading is measured against the zero reference, which is why the
    ZERO button stays reachable during use: heading is the only channel that
    can wander.

WHY THERE IS A ZERO BUTTON AND NOT A MAGNETOMETER
    The BMM150 could give absolute heading, but this probe works around steel:
    engine bays, pipework, machinery. Magnetic heading there is worse than
    gyro drift.

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

# Which body axis the lens looks along. The probe can be mounted any way round,
# so this is chosen once during setup and remembered.
AXES = [
    ("+X", (1.0, 0.0, 0.0)),
    ("-X", (-1.0, 0.0, 0.0)),
    ("+Y", (0.0, 1.0, 0.0)),
    ("-Y", (0.0, -1.0, 0.0)),
    ("+Z", (0.0, 0.0, 1.0)),
    ("-Z", (0.0, 0.0, -1.0)),
]

BG = "#06090b"
PANEL = "#0b1114"
EDGE = "#37474f"
RING = "#3c4a52"
GRID = "#2f6f7a"
BODY = "#7ea8bd"
BODY_EDGE = "#cfe4ef"
LENS = "#0d0d0d"
ARROW = ["#7d8a91", "#98a5ac", "#b4c1c8", "#d2dee4", "#f2f7fa"]
DIM = "#78909c"
MUTED = "#546e7a"


# ------------------------------------------------------------- quaternions

def q_rotate(q, v):
    w, x, y, z = q
    vx, vy, vz = v
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx))


def v_norm(v):
    n = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if n < 1e-9:
        return (0.0, 0.0, 1.0)
    return (v[0] / n, v[1] / n, v[2] / n)


def clamp(x, lo=-1.0, hi=1.0):
    return lo if x < lo else (hi if x > hi else x)


def aim_angles(q_now, q_ref, axis_idx):
    """
    Return (azimuth, elevation) in radians.

    Elevation is measured from gravity, so it is absolute and drift-free.
    Azimuth is measured from the heading captured at zero time, positive to
    the operator's right, because heading has no absolute reference available
    in a steel environment.
    """
    fwd = AXES[axis_idx][1]
    v = v_norm(q_rotate(q_now, fwd))
    el = math.asin(clamp(v[2]))

    if q_ref is None:
        return 0.0, el

    r = v_norm(q_rotate(q_ref, fwd))
    # Signed angle from the reference heading to the current one, about world
    # up. Negated so that turning right reads as positive on screen.
    cross_z = r[0] * v[1] - r[1] * v[0]
    dot = r[0] * v[0] + r[1] * v[1]
    if abs(cross_z) < 1e-12 and abs(dot) < 1e-12:
        return 0.0, el                      # aimed at the zenith: no heading
    return -math.atan2(cross_z, dot), el


# ------------------------------------------------------------- serial link

def find_port():
    for pattern in ("/dev/ttyACM*", "/dev/ttyUSB*"):
        ports = sorted(glob.glob(pattern))
        if ports:
            return ports[0]
    return None


class ProbeLink:
    """
    Demultiplexes the probe's packet stream on a background thread, keeping
    only the newest frame and attitude. Nothing queues: in a live view a late
    frame is worthless, so dropping is the correct policy.
    """

    def __init__(self, port=None, baud=115200):
        self.port = port or find_port()
        self.baud = baud
        self.ser = None
        self.buf = bytearray()
        self.lock = threading.Lock()
        self.frame = None
        self.frame_seq = 0
        self.quat = (1.0, 0.0, 0.0, 0.0)
        self.still = False
        self.status = []
        self.bad_packets = 0
        self.running = False
        self._fps = []

    def start(self):
        if self.port is None:
            raise RuntimeError("No probe found. Check the USB-C cable carries "
                               "data, then run: ls /dev/ttyACM*")
        self.ser = serial.Serial(self.port, self.baud, timeout=0.5)
        time.sleep(2.0)                 # the board reboots when the port opens
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
        self._fps = [t for t in self._fps if now - t < 2.0]
        return len(self._fps) / 2.0

    def _next_packet(self):
        while True:
            idx = self.buf.find(SYNC)
            if idx < 0:
                if len(self.buf) > 1:
                    del self.buf[:-1]       # sync word may straddle two reads
                return None
            if idx > 0:
                del self.buf[:idx]
            if len(self.buf) < 7:
                return None

            ptype = self.buf[2]
            length = int.from_bytes(self.buf[3:7], "little")
            if length > 512 * 1024:         # implausible -> resynced wrongly
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
            except Exception:
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
                            self.quat = tuple(q)
                            self.still = bool(data.get("st"))

                elif ptype == TYPE_FRAME:
                    arr = np.frombuffer(payload, dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        with self.lock:
                            self.frame = img
                            self.frame_seq += 1
                        self._fps.append(time.time())

    def snapshot(self):
        with self.lock:
            return self.frame, self.frame_seq, self.quat, self.still


# ------------------------------------------------------------- indicator

class Indicator:
    """
    Azimuthal projection of the aim direction onto a disc.

    Radius carries the horizontal component, so both straight up and straight
    down collapse to the centre. What separates them is the arrow: it scales
    with elevation, large aiming up and small aiming down, as though the disc
    were being viewed from above with the tip swinging toward or away from you.
    """

    def __init__(self, canvas, cx, cy, radius):
        self.c = canvas
        self.cx, self.cy = cx, cy
        self.R = radius                 # rim = horizontal
        self.items = []
        self.frame_items = []

    def move(self, cx, cy, radius):
        self.cx, self.cy, self.R = cx, cy, radius

    def clear(self):
        for i in self.items:
            self.c.delete(i)
        self.items = []

    def clear_all(self):
        self.clear()
        for i in self.frame_items:
            self.c.delete(i)
        self.frame_items = []

    def draw_frame(self):
        """Rings and cross-hair. Drawn once; the needle redraws on top."""
        for i in self.frame_items:
            self.c.delete(i)
        self.frame_items = []
        c, R = self.c, self.R
        outer = R * 1.30
        for rad, col, w in ((outer, RING, 1.6), (R, RING, 1.2)):
            self.frame_items.append(c.create_oval(
                self.cx - rad, self.cy - rad, self.cx + rad, self.cy + rad,
                outline=col, width=w))
        self.frame_items.append(c.create_line(
            self.cx - outer, self.cy, self.cx + outer, self.cy,
            fill=GRID, width=1))
        self.frame_items.append(c.create_line(
            self.cx, self.cy - outer, self.cx, self.cy + outer,
            fill=GRID, width=1))

    def draw(self, az, el):
        self.clear()
        c, R = self.c, self.R

        # Near-vertical: nothing left to point with, so mark the centre.
        if abs(el) > math.radians(80):
            r = R * 0.13
            up = el > 0
            self.items.append(c.create_oval(
                self.cx - r, self.cy - r, self.cx + r, self.cy + r,
                fill=BODY_EDGE if up else "",
                outline=BODY_EDGE, width=2))
            return

        radial = R * math.cos(el)
        ux, uy = math.sin(az), -math.cos(az)      # screen unit vector, up = 0
        px, py = -uy, ux                          # perpendicular

        # Elevation scale: 1.0 level, ~1.7 at +45, ~0.3 at -45.
        s = clamp(1.0 + math.sin(el), 0.10, 2.0)

        arrow_len = 0.22 * R * s
        arrow_w = 0.13 * R * s
        lens_w = 0.075 * R * max(0.5, 0.75 + 0.25 * s)
        body_w = 0.050 * R * (0.65 + 0.35 * s)
        b0 = 0.04 * R
        gap = 0.008 * R                  # segments read as one object

        def pt(t, off=0.0):
            return (self.cx + ux * t + px * off, self.cy + uy * t + py * off)

        def poly(pts, **kw):
            flat = []
            for p in pts:
                flat += [p[0], p[1]]
            self.items.append(c.create_polygon(*flat, **kw))

        tip = radial
        l1 = tip - arrow_len - gap       # back of the lens block

        # Aiming steeply up leaves almost no radius for the shaft, so the lens
        # block yields space to the body rather than swallowing it whole.
        lens_len = min(0.09 * R * max(0.45, 0.8 * s), max(0.0, (l1 - b0) * 0.55))
        l0 = l1 - lens_len

        if l0 > b0:
            poly([pt(b0, body_w), pt(l0 + gap, body_w),
                  pt(l0 + gap, -body_w), pt(b0, -body_w)],
                 fill=BODY, outline=BODY_EDGE, width=1)

        if lens_len > 0.5:
            # Pure black vanishes against the panel, so the block is outlined.
            poly([pt(l0, lens_w), pt(l1, lens_w),
                  pt(l1, -lens_w), pt(l0, -lens_w)],
                 fill=LENS, outline=BODY_EDGE, width=1)

        # Nested slices give the arrow a gradient without an image layer.
        n = len(ARROW)
        tip_pt = pt(tip)
        for i, g in enumerate(ARROW):
            t0 = a0 + arrow_len * (i / n)
            w = arrow_w * (1.0 - i / n)
            poly([pt(t0, w), pt(t0, -w), tip_pt], fill=g, outline="")


# ------------------------------------------------------------- application

class App:
    STAGE_SETUP = 0      # instructions, no indicator yet
    STAGE_CHECK = 1      # zeroed; verify before entering the live view
    STAGE_RUN = 2

    def __init__(self, link, args):
        self.link = link
        self.args = args
        self.cfg = self.load_cfg()
        self.axis_idx = self.cfg.get("axis", 0)
        self.q_ref = None
        self.stage = self.STAGE_SETUP
        self.last_seq = -1
        self.photo = None

        self.root = tk.Tk()
        self.root.title("Endoscope")
        self.root.configure(bg=BG)
        if args.windowed:
            self.root.geometry("1024x600")
        else:
            self.root.attributes("-fullscreen", True)
            self.root.configure(cursor="none")
        self.root.update_idletasks()

        self.W = (self.root.winfo_width() if args.windowed
                  else self.root.winfo_screenwidth())
        self.H = (self.root.winfo_height() if args.windowed
                  else self.root.winfo_screenheight())

        self.pick_font()
        self.canvas = tk.Canvas(self.root, width=self.W, height=self.H,
                                bg=BG, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)

        self.video_item = self.canvas.create_image(self.W // 2, self.H // 2)
        self.hud = []
        self.stage_items = []
        self.indicator = None

        self.root.bind("<Escape>", lambda e: self.quit())
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

        self.enter_setup()

    # ---- config

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
        have = {f.lower(): f for f in tkfont.families()}
        for name in ("dejavu sans", "liberation sans", "noto sans", "arial"):
            if name in have:
                self.ff = have[name]
                return
        self.ff = "TkDefaultFont"

    def f(self, size, bold=False):
        return (self.ff, size, "bold") if bold else (self.ff, size)

    def text_w(self, s, size, bold=False):
        return tkfont.Font(family=self.ff, size=size,
                           weight="bold" if bold else "normal").measure(s)

    # ---- widgets

    def button(self, cx, cy, label, fill, cb, size, pad_x=26, pad_y=15,
               store=None, anchor="center"):
        """Sized to its own text, so labels can never overlap each other."""
        w = self.text_w(label, size, True) + pad_x * 2
        h = size * 2 + pad_y * 2
        if anchor == "center":
            x0, y0 = cx - w / 2, cy - h / 2
        elif anchor == "nw":
            x0, y0 = cx, cy
        elif anchor == "ne":
            x0, y0 = cx - w, cy
        elif anchor == "se":
            x0, y0 = cx - w, cy - h
        else:
            x0, y0 = cx, cy - h
        r = self.canvas.create_rectangle(x0, y0, x0 + w, y0 + h,
                                         fill=fill, outline="", width=0)
        t = self.canvas.create_text(x0 + w / 2, y0 + h / 2, text=label,
                                    fill="white", font=self.f(size, True))
        for item in (r, t):
            self.canvas.tag_bind(item, "<Button-1>", lambda e: cb())
        if store is not None:
            store.extend([r, t])
        return r, t, w, h

    def clear_stage(self):
        for i in self.stage_items:
            self.canvas.delete(i)
        self.stage_items = []
        if self.indicator:
            self.indicator.clear_all()
            self.indicator = None

    def clear_hud(self):
        for i in self.hud:
            self.canvas.delete(i)
        self.hud = []

    # ---- stage 1: instructions, indicator deliberately absent

    def enter_setup(self):
        self.clear_stage()
        self.clear_hud()
        self.stage = self.STAGE_SETUP
        self.canvas.itemconfigure(self.video_item, image="")
        c, W, H = self.canvas, self.W, self.H
        z = self.stage_items

        big = max(17, int(H / 17))
        mid = max(11, int(H / 34))
        small = max(9, int(H / 44))

        z.append(c.create_text(W / 2, H * 0.11, text="ZERO BEFORE USE",
                               fill="#ffffff", font=self.f(big, True)))
        z.append(c.create_text(
            W / 2, H * 0.215,
            text="Aim the lens straight ahead \u2014 the way you are facing",
            fill="#b0bec5", font=self.f(mid)))
        z.append(c.create_text(
            W / 2, H * 0.275,
            text="Hold it level and still, then press ZERO",
            fill="#b0bec5", font=self.f(mid)))

        # Diagram: viewed from above. The operator is at the bottom, the probe
        # points away from them, which is what the indicator will call "up".
        cx, cy = W / 2, H * 0.53
        span = min(W * 0.22, H * 0.20)
        z.append(c.create_text(cx, cy + span * 0.95, text="YOU",
                               fill=MUTED, font=self.f(small, True)))
        z.append(c.create_oval(cx - span * 0.07, cy + span * 0.55,
                               cx + span * 0.07, cy + span * 0.69,
                               outline=MUTED, width=2))
        z.append(c.create_rectangle(cx - span * 0.30, cy + span * 0.18,
                                    cx + span * 0.30, cy + span * 0.42,
                                    outline=EDGE, width=2))
        z.append(c.create_text(cx + span * 0.62, cy + span * 0.30,
                               text="SCREEN", fill=MUTED,
                               font=self.f(small), anchor="w"))
        z.append(c.create_rectangle(cx - span * 0.055, cy - span * 0.18,
                                    cx + span * 0.055, cy + span * 0.06,
                                    fill=BODY, outline=BODY_EDGE))
        z.append(c.create_rectangle(cx - span * 0.075, cy - span * 0.30,
                                    cx + span * 0.075, cy - span * 0.18,
                                    fill=LENS, outline=LENS))
        z.append(c.create_line(cx, cy - span * 0.36, cx, cy - span * 0.86,
                               fill="#cfd8dc", width=3, arrow="last",
                               arrowshape=(15, 19, 7)))
        z.append(c.create_text(cx, cy - span * 1.00, text="FORWARD",
                               fill="#cfd8dc", font=self.f(small, True)))

        self.button(W / 2, H * 0.855, "ZERO", "#2e7d32", self.do_zero,
                    max(15, int(H / 22)), store=z)
        z.append(c.create_text(
            W / 2, H * 0.955,
            text="lens axis: " + AXES[self.axis_idx][0],
            fill=MUTED, font=self.f(small)))

    def do_zero(self):
        _, _, q, _ = self.link.snapshot()
        self.q_ref = q
        self.enter_check()

    # ---- stage 2: verify the zero before the video takes over

    def enter_check(self):
        self.clear_stage()
        self.stage = self.STAGE_CHECK
        c, W, H = self.canvas, self.W, self.H
        z = self.stage_items

        big = max(15, int(H / 21))
        small = max(9, int(H / 44))

        z.append(c.create_text(W / 2, H * 0.085, text="CHECK THE ZERO",
                               fill="#ffffff", font=self.f(big, True)))

        disc = min(W, H) * 0.24
        self.indicator = Indicator(c, W / 2, H * 0.44, disc)
        self.indicator.draw_frame()

        self.check_readout = c.create_text(
            W / 2, H * 0.44 + disc * 1.55, text="",
            fill=DIM, font=self.f(small))
        z.append(self.check_readout)

        z.append(c.create_text(
            W / 2, H * 0.735,
            text="Pointing forward it should read straight up.",
            fill="#b0bec5", font=self.f(small)))
        z.append(c.create_text(
            W / 2, H * 0.785,
            text="Tilt the lens up: the arrow grows.   Turn right: it swings right.",
            fill=MUTED, font=self.f(small)))
        z.append(c.create_text(
            W / 2, H * 0.832,
            text="If it moves the wrong way, press AXIS and zero again.",
            fill=MUTED, font=self.f(small)))

        bs = max(13, int(H / 26))
        gap = max(14, int(W / 60))
        labels = [("AXIS " + AXES[self.axis_idx][0], "#455a64", self.cycle_axis),
                  ("RE-ZERO", "#1565c0", self.do_zero),
                  ("START", "#2e7d32", self.enter_run)]
        widths = [self.text_w(t, bs, True) + 52 for t, _, _ in labels]
        total = sum(widths) + gap * (len(widths) - 1)
        x = W / 2 - total / 2
        self.axis_btn = None
        for (label, col, cb), w in zip(labels, widths):
            r, t, _, _ = self.button(x + w / 2, H * 0.925, label, col, cb, bs,
                                     store=z)
            if label.startswith("AXIS"):
                self.axis_btn = t
            x += w + gap

    def cycle_axis(self):
        self.axis_idx = (self.axis_idx + 1) % len(AXES)
        self.save_cfg()
        # A new axis invalidates the old reference, so re-zero immediately.
        _, _, q, _ = self.link.snapshot()
        self.q_ref = q
        self.enter_check()

    # ---- stage 3: live view

    def enter_run(self):
        self.clear_stage()
        self.stage = self.STAGE_RUN
        c, W, H = self.canvas, self.W, self.H
        pad = max(10, H // 46)
        bs = max(12, int(H / 30))

        self.button(pad, pad, "\u2715  EXIT", "#c62828", self.quit, bs,
                    store=self.hud, anchor="nw")
        self.button(W - pad, H - pad, "ZERO", "#1565c0", self.do_zero, bs,
                    store=self.hud, anchor="se")

        panel = int(min(W, H) * 0.33)
        px, py = W - panel - pad, pad
        self.hud.append(c.create_rectangle(px, py, px + panel, py + panel,
                                           fill=PANEL, outline=EDGE, width=2))
        disc = panel * 0.33
        self.indicator = Indicator(c, px + panel / 2,
                                   py + panel * 0.46, disc)
        self.indicator.draw_frame()
        self.hud.extend(self.indicator.frame_items)

        self.readout = c.create_text(px + panel / 2, py + panel * 0.90,
                                     text="", fill=DIM,
                                     font=self.f(max(9, panel // 17)))
        self.hud.append(self.readout)
        self.statusbar = c.create_text(pad, H - pad, anchor="sw", text="",
                                       fill=MUTED,
                                       font=self.f(max(8, H // 62)))
        self.hud.append(self.statusbar)

    # ---- loop

    def update(self):
        frame, seq, q, still = self.link.snapshot()
        az, el = aim_angles(q, self.q_ref, self.axis_idx)

        if self.stage == self.STAGE_RUN:
            if frame is not None and seq != self.last_seq:
                self.last_seq = seq
                h, w = frame.shape[:2]
                scale = min(self.W / w, self.H / h)
                small = cv2.resize(frame, (int(w * scale), int(h * scale)),
                                   interpolation=cv2.INTER_LINEAR)
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
                self.canvas.itemconfigure(self.video_item, image=self.photo)
                self.canvas.tag_lower(self.video_item)
                for i in self.hud:
                    self.canvas.tag_raise(i)

            self.indicator.draw(az, el)
            self.canvas.itemconfigure(
                self.readout,
                text="AZ {:+.0f}\u00b0   EL {:+.0f}\u00b0".format(
                    math.degrees(az), math.degrees(el)))
            self.canvas.itemconfigure(
                self.statusbar,
                text="{:.0f} fps    {}".format(
                    self.link.video_fps(),
                    "re-calibrating" if still else ""))

        elif self.stage == self.STAGE_CHECK:
            self.indicator.draw(az, el)
            self.canvas.itemconfigure(
                self.check_readout,
                text="AZ {:+.0f}\u00b0   EL {:+.0f}\u00b0".format(
                    math.degrees(az), math.degrees(el)))

        self.root.after(40, self.update)

    def quit(self):
        try:
            self.root.destroy()
        except Exception:
            pass

    def run(self):
        self.root.after(120, self.update)
        self.root.mainloop()


def main():
    ap = argparse.ArgumentParser(description="Endoscope viewer with aim indicator")
    ap.add_argument("--port", help="e.g. /dev/ttyACM0 (auto-detected if omitted)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--windowed", action="store_true")
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
