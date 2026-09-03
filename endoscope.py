#!/usr/bin/env python3
"""
Endoscope viewer with a 3D aim indicator  (v3)
----------------------------------------------
Fullscreen video from the AtomS3R-CAM probe plus a compass showing where the
lens is aimed, for work where the probe is out of sight.

    DISPLAY=:0 python3 endoscope.py
    python3 endoscope.py --windowed         # development on a desktop
    python3 endoscope.py --sim --windowed   # no hardware at all: synthetic probe
    python3 endoscope.py --port /dev/ttyACM0
    python3 endoscope.py --log drift.csv    # record az/el vs time for analysis

KEYS (besides the touch buttons): Z = zero, A = axis, S = start, Esc = exit.

LAYOUT
    top-left      EXIT (every stage -- a touchscreen has no Escape key)
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
    can wander. In the live view ZERO re-captures the reference in place --
    aim forward, tap it, keep working.

WHY THERE IS A ZERO BUTTON AND NOT A MAGNETOMETER
    The BMM150 could give absolute heading, but this probe works around steel:
    engine bays, pipework, machinery. Magnetic heading there is worse than
    gyro drift.

WHAT v3 FIXES OVER v2 (v2 was written but never ran)
    - Indicator.draw() crashed with NameError on an undefined 'a0' the first
      time it rendered a needle. Fixed (arrow base = tip - arrow_len).
    - Stage changes leaked canvas items: ZERO pressed in the live view drew
      the check screen on top of the still-visible video and HUD. All
      transitions now run through set_stage(), which clears everything.
    - The serial link now connects and RECONNECTS by itself and the UI shows
      its state. The probe is USB-powered, so an unplug is a reboot: the
      quaternion restarts and any old zero is garbage. The app detects the
      restart and forces a re-zero instead of silently pointing wrong.
    - Startup no longer flushes the probe's status packets. ZERO is disabled
      (grey) until real attitude data is flowing, so a zero can no longer be
      captured from the identity quaternion.
    - Zeroing while steeply tilted leaves almost no horizontal reference for
      heading; the check screen now warns and asks for a level re-zero.
    - The check screen shows elapsed-time-since-zero and the peak heading
      excursion: leave the probe still and that number IS the drift, which is
      the measurement that decides whether this product works (HANDOFF §9.3).
    - Layout is computed from a running y-cursor instead of magic fractions;
      on the actual 1024x600 panel v2's check screen overlapped itself.
    - --sim runs the entire app against a synthetic probe.

SETUP ON THE PI
    sudo apt install python3-serial python3-opencv python3-pil.imagetk python3-numpy -y
    sudo usermod -aG dialout $USER      # then REBOOT
"""
import argparse
import csv
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

try:
    import numpy as np
except ImportError:
    print("numpy missing. Run: sudo apt install python3-numpy -y",
          file=sys.stderr)
    sys.exit(1)

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

# Corrupted-length guard, per packet type. A corrupted length below the cap
# makes the parser sit waiting for bytes that never come, so the caps are as
# tight as the real traffic allows: IMU JSON is ~200 bytes, frames ~10-40 KB.
MAX_LEN = {TYPE_IMU: 4 * 1024, TYPE_FRAME: 256 * 1024}

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
WARN = "#ffb300"
ALERT = "#ef5350"
OK = "#66bb6a"


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


def q_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def v_norm(v):
    n = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if n < 1e-9:
        return (0.0, 0.0, 1.0)
    return (v[0] / n, v[1] / n, v[2] / n)


def clamp(x, lo=-1.0, hi=1.0):
    return lo if x < lo else (hi if x > hi else x)


# Below this horizontal magnitude the heading of a vector is numerically
# meaningless (the lens is within ~0.06 deg of vertical); atan2 of noise would
# spin the needle. asin(1e-3) is far below anything the indicator can show.
H_EPS = 1e-3


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
    # Heading lives in the horizontal projections. If either vector is
    # (numerically) vertical there is no heading to compare -- report zero
    # rather than the atan2 of noise.
    rh = math.hypot(r[0], r[1])
    vh = math.hypot(v[0], v[1])
    if rh < H_EPS or vh < H_EPS:
        return 0.0, el

    # Signed angle from the reference heading to the current one, about world
    # up. Negated so that turning right reads as positive on screen.
    cross_z = r[0] * v[1] - r[1] * v[0]
    dot = r[0] * v[0] + r[1] * v[1]
    return -math.atan2(cross_z, dot), el


def ref_elevation(q_ref, axis_idx):
    """Elevation the reference was captured at; used to warn on steep zeros."""
    if q_ref is None:
        return 0.0
    v = v_norm(q_rotate(q_ref, AXES[axis_idx][1]))
    return math.asin(clamp(v[2]))


# ------------------------------------------------------------- serial link

def find_port(preferred=None):
    if preferred:
        return preferred if os.path.exists(preferred) else None
    for pattern in ("/dev/ttyACM*", "/dev/ttyUSB*"):
        ports = sorted(glob.glob(pattern))
        if ports:
            return ports[0]
    return None


class ProbeLink:
    """
    Owns the USB connection on a background thread: finds the port, opens it,
    demultiplexes the packet stream, and when the link dies -- cable pulled,
    board rebooted, port vanished -- closes, waits, and reconnects by itself.
    The UI never blocks on any of this; it just reads the published state.

    Only the newest frame and attitude are kept. Nothing queues: in a live
    view a late frame is worthless, so dropping is the correct policy.

    `generation` increments every time the probe (re)connects or announces a
    fresh boot. The probe is USB-powered, so losing the link means it lost
    power and its orientation estimate restarted from scratch -- any zero
    reference captured before that is garbage. The app compares generations
    to know when to force a re-zero.
    """

    def __init__(self, port=None, baud=115200):
        self.want_port = port
        self.baud = baud
        self.port = None
        self.ser = None
        self.buf = bytearray()
        self.lock = threading.Lock()

        self.frame = None
        self.frame_seq = 0
        self.quat = (1.0, 0.0, 0.0, 0.0)
        self.still = False

        self.state = "searching"        # searching / connecting / online / offline
        self.fw = ""                    # last firmware status string
        self.calibrating = False
        self.camera_failed = False
        self.generation = 0
        self.imu_time = 0.0
        self.frame_time = 0.0
        self.status = []
        self.bad_packets = 0
        self._fps = []
        self.running = False

    # ---- lifecycle

    def start(self):
        self.running = True
        threading.Thread(target=self._manager, daemon=True).start()
        return self

    def stop(self):
        self.running = False
        ser = self.ser
        if ser:
            try:
                ser.close()             # unblocks the read
            except Exception:
                pass

    def _manager(self):
        while self.running:
            port = find_port(self.want_port)
            if port is None:
                self._set_state("searching")
                time.sleep(1.0)
                continue
            self._set_state("connecting")
            try:
                # Opening the port asserts DTR/RTS, which resets the board:
                # expect boot-ROM text before our packets. The parser's sync
                # scan skips it, so nothing is flushed -- flushing is how v2
                # lost every startup status message.
                self.ser = serial.Serial(port, self.baud, timeout=0.5)
            except Exception:
                self._set_state("offline")
                time.sleep(1.5)
                continue
            self.port = port
            self.buf.clear()
            with self.lock:
                self.generation += 1
            self._set_state("online")
            self._read_until_error()
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None
            self._set_state("offline")
            time.sleep(1.5)

    def _set_state(self, s):
        with self.lock:
            self.state = s

    # ---- receive path

    def _read_until_error(self):
        while self.running:
            try:
                chunk = self.ser.read(self.ser.in_waiting or 1)
            except Exception:
                return
            if chunk:
                self.buf.extend(chunk)
                # Boot garbage can pile up ahead of the first sync word.
                if len(self.buf) > 1024 * 1024:
                    del self.buf[:len(self.buf) - 4096]
            while True:
                pkt = self._next_packet()
                if pkt is None:
                    break
                self._handle(*pkt)

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
            if length > MAX_LEN.get(ptype, 0):  # junk type or absurd length:
                del self.buf[:2]                # resynced wrongly, skip sync
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

    def _handle(self, ptype, payload):
        if ptype == TYPE_IMU:
            try:
                data = json.loads(payload.decode("utf-8", "ignore"))
            except json.JSONDecodeError:
                return
            if "status" in data:
                self._handle_status(data)
                return
            q = data.get("q")
            if q and len(q) == 4:
                with self.lock:
                    self.quat = tuple(q)
                    self.still = bool(data.get("st"))
                    self.imu_time = time.monotonic()

        elif ptype == TYPE_FRAME:
            arr = np.frombuffer(payload, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                now = time.monotonic()
                with self.lock:
                    self.frame = img
                    self.frame_seq += 1
                    self.frame_time = now
                    self._fps.append(now)

    def _handle_status(self, data):
        st = str(data.get("status", ""))
        with self.lock:
            self.status.append(data)
            if len(self.status) > 50:
                del self.status[:-50]
            self.fw = st
            if st in ("camera_ok", "camera_failed"):
                # Boot banner: the firmware only prints this once per power-up,
                # so seeing it again means the probe restarted.
                self.generation += 1
                self.camera_failed = (st == "camera_failed")
            elif st == "calibrating":
                self.calibrating = True
            elif st in ("calibrated", "calib_moved", "ready"):
                self.calibrating = False

    # ---- what the UI reads

    def snapshot(self):
        with self.lock:
            return self.frame, self.frame_seq, self.quat, self.still

    def health(self):
        now = time.monotonic()
        with self.lock:
            self._fps = [t for t in self._fps if now - t < 2.0]
            return {
                "state": self.state,
                "fw": self.fw,
                "calibrating": self.calibrating,
                "camera_failed": self.camera_failed,
                "generation": self.generation,
                "imu_age": now - self.imu_time if self.imu_time else 1e9,
                "frame_age": now - self.frame_time if self.frame_time else 1e9,
                "fps": len(self._fps) / 2.0,
                "bad": self.bad_packets,
                "port": self.port,
            }


class SimLink:
    """
    Synthetic probe with the same interface as ProbeLink: a scripted attitude
    (slow heading sweep, occasional tilt, pauses) and a generated test-pattern
    video. Lets the whole app -- stages, indicator, HUD, drift readout -- run
    and be exercised on any machine with no hardware attached.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.frame_seq = 0
        self.quat = (1.0, 0.0, 0.0, 0.0)
        self.still = False
        self.state = "online"
        self.fw = "ready"
        self.calibrating = False
        self.camera_failed = False
        self.generation = 1
        self.imu_time = 0.0
        self.frame_time = 0.0
        self.status = [{"status": "camera_ok", "pid": "0xSIM"},
                       {"status": "imu_ok"}, {"status": "ready"}]
        self.bad_packets = 0
        self.running = False
        self.t0 = time.monotonic()

    def start(self):
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()
        return self

    def stop(self):
        self.running = False

    def _pose(self, t):
        # Heading wanders +/-50 deg; pitch breathes +/-30 deg; roll stays 0.
        yaw = math.radians(50.0 * math.sin(2 * math.pi * t / 24.0))
        pitch = math.radians(30.0 * math.sin(2 * math.pi * t / 9.0))
        qz = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        qy = (math.cos(pitch / 2), 0.0, -math.sin(pitch / 2), 0.0)
        return q_mul(qz, qy)            # lens along +X, nose-up positive

    def _loop(self):
        frame_due = 0.0
        while self.running:
            t = time.monotonic() - self.t0
            with self.lock:
                self.quat = self._pose(t)
                self.still = (int(t) % 12) < 2
                self.imu_time = time.monotonic()
            if t >= frame_due:
                frame_due = t + 0.08
                img = np.zeros((240, 320, 3), np.uint8)
                img[:] = (28, 20, 12)
                x = int(160 + 110 * math.sin(t * 0.9))
                y = int(120 + 70 * math.cos(t * 0.6))
                cv2.circle(img, (x, y), 26, (60, 160, 240), -1)
                cv2.putText(img, "SIM %5.1fs" % t, (10, 228),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 230), 1)
                with self.lock:
                    self.frame = img
                    self.frame_seq += 1
                    self.frame_time = time.monotonic()
            time.sleep(0.02)

    def snapshot(self):
        with self.lock:
            return self.frame, self.frame_seq, self.quat, self.still

    def health(self):
        now = time.monotonic()
        with self.lock:
            return {"state": "online", "fw": self.fw, "calibrating": False,
                    "camera_failed": False, "generation": self.generation,
                    "imu_age": now - self.imu_time if self.imu_time else 1e9,
                    "frame_age": now - self.frame_time if self.frame_time else 1e9,
                    "fps": 12.5, "bad": 0, "port": "sim"}


# ------------------------------------------------------------- indicator

class Indicator:
    """
    Azimuthal projection of the aim direction onto a disc.

    Radius carries the horizontal component, so both straight up and straight
    down collapse to the centre. What separates them is the arrow: it scales
    with elevation, large aiming up and small aiming down, as though the disc
    were being viewed from above with the tip swinging toward or away from you.

    The drawn needle follows a lightly smoothed copy of the projected point
    (a wrap-free 2D EMA), because near the vertical the heading of the raw
    vector is dominated by noise and the needle would spin.
    """

    POLE = math.radians(80)

    def __init__(self, canvas, cx, cy, radius):
        self.c = canvas
        self.cx, self.cy = cx, cy
        self.R = radius                 # rim = horizontal
        self.items = []
        self.frame_items = []
        self._sm = None                 # smoothed (x, y, scale)

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
        if abs(el) > self.POLE:
            self._sm = None
            r = R * 0.13
            up = el > 0
            self.items.append(c.create_oval(
                self.cx - r, self.cy - r, self.cx + r, self.cy + r,
                fill=BODY_EDGE if up else "",
                outline=BODY_EDGE, width=2))
            return

        # Raw projected point and elevation scale...
        radial = R * math.cos(el)
        tx = math.sin(az) * radial
        ty = -math.cos(az) * radial               # canvas y grows downward
        ts = clamp(1.0 + math.sin(el), 0.10, 2.0)  # 1.0 level, ~1.7 up45, ~0.3 dn45

        # ...smoothed in 2D, which is wrap-free (an angle EMA would glitch
        # crossing +/-180). Alpha 0.45 at 25 fps ~= 70 ms lag: invisible in
        # use, enough to stop the near-vertical spin.
        if self._sm is None:
            self._sm = [tx, ty, ts]
        else:
            a = 0.45
            self._sm[0] += a * (tx - self._sm[0])
            self._sm[1] += a * (ty - self._sm[1])
            self._sm[2] += a * (ts - self._sm[2])
        sx, sy, s = self._sm
        radial = min(math.hypot(sx, sy), R)
        if radial > 1e-6:
            ux, uy = sx / max(radial, 1e-9), sy / max(radial, 1e-9)
        else:
            ux, uy = math.sin(az), -math.cos(az)
        px, py = -uy, ux                          # perpendicular

        tip = radial
        # Cap the arrow so it cannot shoot out past the centre to the far
        # side of the disc when steeply elevated (radial small, scale large).
        # It converges toward the centre dot instead, which is where the
        # display is about to go anyway.
        b0 = 0.04 * R
        arrow_len = min(0.22 * R * s, max(tip - b0, 0.06 * R))
        arrow_w = min(0.13 * R * s, arrow_len * 1.15)
        lens_w = 0.075 * R * max(0.5, 0.75 + 0.25 * s)
        body_w = 0.050 * R * (0.65 + 0.35 * s)
        gap = 0.008 * R                  # segments read as one object

        def pt(t, off=0.0):
            return (self.cx + ux * t + px * off, self.cy + uy * t + py * off)

        def poly(pts, **kw):
            flat = []
            for p in pts:
                flat += [p[0], p[1]]
            self.items.append(c.create_polygon(*flat, **kw))

        a0 = tip - arrow_len             # arrowhead base
        l1 = a0 - gap                    # front of the lens block

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
        self.axis_idx = int(self.cfg.get("axis", 0)) % len(AXES)
        self.q_ref = None
        self.zero_gen = -1              # link generation the zero belongs to
        self.zero_time = 0.0
        self.peak_az = 0.0
        self.stage = self.STAGE_SETUP
        self.last_seq = -1
        self.photo = None
        self.notice = ""                # red line on the setup screen
        self.toast_items = []
        self.toast_after = None
        self.log = self._open_log(args.log) if args.log else None
        self._log_rows = 0

        self.root = tk.Tk()
        self.root.title("Endoscope")
        self.root.configure(bg=BG)
        if args.windowed:
            self.root.geometry("1024x600")
            self.root.update()          # map it so winfo returns real numbers
            self.W = self.root.winfo_width()
            self.H = self.root.winfo_height()
        else:
            self.root.attributes("-fullscreen", True)
            self.root.configure(cursor="none")
            self.root.update_idletasks()
            self.W = self.root.winfo_screenwidth()
            self.H = self.root.winfo_screenheight()

        self.pick_font()
        self._fonts = {}
        self.canvas = tk.Canvas(self.root, width=self.W, height=self.H,
                                bg=BG, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)

        self.video_item = self.canvas.create_image(self.W // 2, self.H // 2)
        self.hud = []
        self.stage_items = []
        self.indicator = None
        self.readout = None
        self.statusbar = None
        self.check_readout = None
        self.drift_line = None
        self.setup_status = None
        self.zero_btn = None            # (rect, text) of the setup ZERO
        self.nosignal = None

        self.root.bind("<Escape>", lambda e: self.quit())
        self.root.bind("<Key>", self.on_key)
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

        self.set_stage(self.STAGE_SETUP)

    # ---- config / logging

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

    def _open_log(self, path):
        f = open(path, "w", newline="")
        w = csv.writer(f)
        w.writerow(["t_s", "stage", "az_deg", "el_deg",
                    "qw", "qx", "qy", "qz", "still"])
        return (f, w, time.monotonic())

    def _log_row(self, az, el, q, still):
        if not self.log:
            return
        f, w, t0 = self.log
        w.writerow(["%.3f" % (time.monotonic() - t0), self.stage,
                    "%.2f" % math.degrees(az), "%.2f" % math.degrees(el),
                    "%.5f" % q[0], "%.5f" % q[1], "%.5f" % q[2], "%.5f" % q[3],
                    int(still)])
        self._log_rows += 1
        if self._log_rows % 50 == 0:
            f.flush()

    # ---- fonts (cached: per-call tkfont.Font objects crash the process,
    #      see v2's hard-won note -- Font.__del__ calls into Tcl and the GC
    #      may run it on the serial thread, corrupting Tcl's allocator)

    def pick_font(self):
        have = {f.lower(): f for f in tkfont.families()}
        for name in ("dejavu sans", "liberation sans", "noto sans", "arial"):
            if name in have:
                self.ff = have[name]
                return
        self.ff = "TkDefaultFont"

    def f(self, size, bold=False):
        return (self.ff, size, "bold") if bold else (self.ff, size)

    def font_obj(self, size, bold=False):
        key = (size, bold)
        fnt = self._fonts.get(key)
        if fnt is None:
            fnt = tkfont.Font(family=self.ff, size=size,
                              weight="bold" if bold else "normal")
            self._fonts[key] = fnt
        return fnt

    def text_w(self, s, size, bold=False):
        return self.font_obj(size, bold).measure(s)

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

    def toast(self, text, color="#ffffff", ms=900):
        for i in self.toast_items:
            self.canvas.delete(i)
        self.toast_items = []
        if self.toast_after:
            self.root.after_cancel(self.toast_after)
            self.toast_after = None
        size = max(13, int(self.H / 26))
        w = self.text_w(text, size, True) + 60
        x, y = self.W / 2, self.H * 0.78
        self.toast_items.append(self.canvas.create_rectangle(
            x - w / 2, y - size - 14, x + w / 2, y + size + 14,
            fill="#111a1f", outline=EDGE, width=1))
        self.toast_items.append(self.canvas.create_text(
            x, y, text=text, fill=color, font=self.f(size, True)))
        self.toast_after = self.root.after(ms, self._toast_off)

    def _toast_off(self):
        for i in self.toast_items:
            self.canvas.delete(i)
        self.toast_items = []
        self.toast_after = None

    # ---- stage machinery: every transition goes through here, and it clears
    #      EVERYTHING. v2 had two item stores cleared in different places,
    #      which is how pressing ZERO in the live view stacked the check
    #      screen on top of the still-visible video and HUD.

    def set_stage(self, stage):
        for i in self.stage_items:
            self.canvas.delete(i)
        self.stage_items = []
        for i in self.hud:
            self.canvas.delete(i)
        self.hud = []
        if self.indicator:
            self.indicator.clear_all()
            self.indicator = None
        self._toast_off()
        self.readout = None
        self.statusbar = None
        self.check_readout = None
        self.drift_line = None
        self.setup_status = None
        self.zero_btn = None
        self.nosignal = None
        if stage != self.STAGE_RUN:
            self.canvas.itemconfigure(self.video_item, image="")
            self.photo = None
            self.last_seq = -1

        self.stage = stage
        if stage == self.STAGE_SETUP:
            self.enter_setup()
        elif stage == self.STAGE_CHECK:
            self.enter_check()
        else:
            self.enter_run()

    # ---- zero capture, shared by every stage

    def link_ready(self):
        h = self.link.health()
        return h["state"] == "online" and h["imu_age"] < 1.0

    def capture_zero(self):
        if not self.link_ready():
            self.toast("PROBE NOT READY", WARN)
            return False
        _, _, q, _ = self.link.snapshot()
        self.q_ref = q
        self.zero_gen = self.link.health()["generation"]
        self.zero_time = time.monotonic()
        self.peak_az = 0.0
        return True

    # ---- stage 1: instructions, indicator deliberately absent
    #      (explicit user requirement: no indicator until after ZERO)

    def enter_setup(self):
        c, W, H = self.canvas, self.W, self.H
        z = self.stage_items

        big = max(17, int(H / 17))
        mid = max(11, int(H / 34))
        small = max(9, int(H / 44))

        self.button(max(10, H // 46), max(10, H // 46), "\u2715  EXIT",
                    "#c62828", self.quit, max(11, int(H / 40)), store=z,
                    anchor="nw")

        z.append(c.create_text(W / 2, H * 0.10, text="ZERO BEFORE USE",
                               fill="#ffffff", font=self.f(big, True)))
        if self.notice:
            z.append(c.create_text(W / 2, H * 0.165, text=self.notice,
                                   fill=ALERT, font=self.f(mid, True)))
        z.append(c.create_text(
            W / 2, H * 0.225,
            text="Aim the lens straight ahead \u2014 the way you are facing",
            fill="#b0bec5", font=self.f(mid)))
        z.append(c.create_text(
            W / 2, H * 0.285,
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

        r, t, _, _ = self.button(W / 2, H * 0.855, "ZERO", MUTED,
                                 self.do_zero, max(15, int(H / 22)), store=z)
        self.zero_btn = (r, t)          # recoloured live once the probe is up
        z.append(c.create_text(
            W / 2, H * 0.955,
            text="lens axis: " + AXES[self.axis_idx][0],
            fill=MUTED, font=self.f(small)))
        self.setup_status = c.create_text(
            max(10, H // 46), H - max(10, H // 46), anchor="sw",
            text="PROBE: \u2026", fill=DIM, font=self.f(small))
        z.append(self.setup_status)

    def do_zero(self):
        if not self.capture_zero():
            return
        self.notice = ""
        self.set_stage(self.STAGE_CHECK)

    # ---- stage 2: verify the zero before the video takes over
    #      (explicit user requirement: must not jump straight to video)

    def enter_check(self):
        c, W, H = self.canvas, self.W, self.H
        z = self.stage_items
        pad = max(10, H // 46)

        big = max(15, int(H / 22))
        small = max(9, int(H / 42))
        tiny = max(9, int(H / 48))

        self.button(pad, pad, "\u2715  EXIT", "#c62828", self.quit,
                    max(11, int(H / 40)), store=z, anchor="nw")

        y = H * 0.058
        z.append(c.create_text(W / 2, y, text="CHECK THE ZERO",
                               fill="#ffffff", font=self.f(big, True)))

        # Disc, sized so the outer ring plus every text line below it stacks
        # inside the screen. v2 placed these with independent magic fractions
        # and they overlapped on the real 1024x600 panel.
        R = min(W, H) * 0.185
        cy = y + big + R * 1.30 + H * 0.015
        self.indicator = Indicator(c, W / 2, cy, R)
        self.indicator.draw_frame()
        y = cy + R * 1.30 + H * 0.012

        y += small
        self.check_readout = c.create_text(W / 2, y, text="",
                                           fill="#eceff1",
                                           font=self.f(small, True))
        z.append(self.check_readout)

        y += small + tiny + H * 0.012
        self.drift_line = c.create_text(W / 2, y, text="", fill=DIM,
                                        font=self.f(tiny))
        z.append(self.drift_line)

        y += tiny * 2 + H * 0.018
        z.append(c.create_text(
            W / 2, y,
            text="Tilt the lens up \u2014 the arrow grows.   Turn right \u2014 it swings right.",
            fill="#b0bec5", font=self.f(tiny)))
        y += tiny * 2 + H * 0.006
        z.append(c.create_text(
            W / 2, y,
            text="Moves the wrong way? Press AXIS, then it re-zeros itself.",
            fill=MUTED, font=self.f(tiny)))

        if abs(ref_elevation(self.q_ref, self.axis_idx)) > math.radians(60):
            y += tiny * 2 + H * 0.006
            z.append(c.create_text(
                W / 2, y,
                text="\u26a0  Zeroed while steeply tilted \u2014 heading reference is weak. "
                     "Aim level and RE-ZERO.",
                fill=WARN, font=self.f(tiny, True)))

        bs = max(13, int(H / 27))
        gap = max(14, int(W / 60))
        labels = [("AXIS " + AXES[self.axis_idx][0], "#455a64", self.cycle_axis),
                  ("RE-ZERO", "#1565c0", self.do_zero),
                  ("START", "#2e7d32", self.start_run)]
        widths = [self.text_w(t, bs, True) + 52 for t, _, _ in labels]
        total = sum(widths) + gap * (len(widths) - 1)
        x = W / 2 - total / 2
        by = H - pad - (bs * 2 + 30) / 2
        for (label, col, cb), w in zip(labels, widths):
            self.button(x + w / 2, by, label, col, cb, bs, store=z)
            x += w + gap

    def cycle_axis(self):
        self.axis_idx = (self.axis_idx + 1) % len(AXES)
        self.save_cfg()
        # A new axis invalidates the old reference, so re-zero immediately.
        if self.capture_zero():
            self.set_stage(self.STAGE_CHECK)

    def start_run(self):
        self.set_stage(self.STAGE_RUN)

    # ---- stage 3: live view

    def enter_run(self):
        c, W, H = self.canvas, self.W, self.H
        pad = max(10, H // 46)
        bs = max(12, int(H / 30))

        self.button(pad, pad, "\u2715  EXIT", "#c62828", self.quit, bs,
                    store=self.hud, anchor="nw")
        self.button(W - pad, H - pad, "ZERO", "#1565c0", self.zero_inplace,
                    bs, store=self.hud, anchor="se")

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
        self.nosignal = c.create_text(W / 2, H / 2, text="NO VIDEO SIGNAL",
                                      fill=MUTED,
                                      font=self.f(max(15, int(H / 20)), True),
                                      state="hidden")
        self.hud.append(self.nosignal)

    def zero_inplace(self):
        """
        Re-zero without leaving the video. Heading drifts; the fix is one tap:
        aim forward, press ZERO, keep working. Going back through the check
        screen mid-inspection would only make the honest mitigation annoying.
        """
        if self.capture_zero():
            self.toast("ZEROED \u2713", OK)

    # ---- keyboard shortcuts (development convenience; the device is touch)

    def on_key(self, e):
        k = e.keysym.lower()
        if k == "z":
            if self.stage == self.STAGE_RUN:
                self.zero_inplace()
            else:
                self.do_zero()
        elif k == "a" and self.stage == self.STAGE_CHECK:
            self.cycle_axis()
        elif k == "s" and self.stage == self.STAGE_CHECK:
            self.start_run()

    # ---- per-frame update

    def probe_status_text(self, h):
        if h["state"] == "searching":
            return "PROBE: searching for USB device\u2026", DIM
        if h["state"] == "connecting":
            return "PROBE: connecting\u2026", DIM
        if h["state"] == "offline":
            return "PROBE: disconnected \u2014 retrying\u2026", ALERT
        # online:
        if h["calibrating"]:
            return "PROBE: calibrating gyro \u2014 keep it still\u2026", WARN
        if h["imu_age"] > 1.0:
            return "PROBE: waiting for data\u2026", DIM
        s = "PROBE: online"
        if h["camera_failed"]:
            s += " \u2014 IMU ok, CAMERA FAILED (see HANDOFF \u00a73)"
            return s, WARN
        if h["fw"] == "calib_moved":
            s += " \u2014 probe moved during calibration; keep it still a moment"
            return s, WARN
        return s, OK

    def update(self):
        frame, seq, q, still = self.link.snapshot()
        h = self.link.health()
        az, el = aim_angles(q, self.q_ref, self.axis_idx)
        self._log_row(az, el, q, still)

        # A probe restart (new generation) restarts its orientation estimate,
        # so any zero captured before it is meaningless. Never keep pointing
        # with a stale reference -- go back and say why.
        if (self.stage != self.STAGE_SETUP and self.q_ref is not None
                and h["generation"] != self.zero_gen):
            self.q_ref = None
            self.notice = "PROBE RESTARTED \u2014 ZERO AGAIN"
            self.set_stage(self.STAGE_SETUP)
            self.root.after(40, self.update)
            return

        if self.stage == self.STAGE_SETUP:
            if self.setup_status:
                text, col = self.probe_status_text(h)
                self.canvas.itemconfigure(self.setup_status,
                                          text=text, fill=col)
            if self.zero_btn:
                ready = h["state"] == "online" and h["imu_age"] < 1.0
                self.canvas.itemconfigure(self.zero_btn[0],
                                          fill="#2e7d32" if ready else MUTED)

        elif self.stage == self.STAGE_CHECK:
            self.peak_az = max(self.peak_az, abs(math.degrees(az)))
            self.indicator.draw(az, el)
            self.canvas.itemconfigure(
                self.check_readout,
                text="AZ {:+.0f}\u00b0   EL {:+.0f}\u00b0".format(
                    math.degrees(az), math.degrees(el)))
            dt = time.monotonic() - self.zero_time
            self.canvas.itemconfigure(
                self.drift_line,
                text="since zero {:d}:{:02d}   AZ peak \u00b1{:.1f}\u00b0"
                     "     (leave it still and the peak is the drift)".format(
                         int(dt) // 60, int(dt) % 60, self.peak_az))

        elif self.stage == self.STAGE_RUN:
            if frame is not None and seq != self.last_seq:
                self.last_seq = seq
                fh, fw = frame.shape[:2]
                scale = min(self.W / fw, self.H / fh)
                small = cv2.resize(frame, (int(fw * scale), int(fh * scale)),
                                   interpolation=cv2.INTER_LINEAR)
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
                self.canvas.itemconfigure(self.video_item, image=self.photo)
                self.canvas.tag_lower(self.video_item)
                for i in self.hud:
                    self.canvas.tag_raise(i)

            stale = h["frame_age"] > 2.5
            self.canvas.itemconfigure(
                self.nosignal,
                state="normal" if stale else "hidden",
                text=("PROBE DISCONNECTED" if h["state"] != "online"
                      else "NO VIDEO SIGNAL"))
            if stale and self.photo is not None:
                self.canvas.itemconfigure(self.video_item, image="")
                self.photo = None

            self.indicator.draw(az, el)
            self.canvas.itemconfigure(
                self.readout,
                text="AZ {:+.0f}\u00b0   EL {:+.0f}\u00b0".format(
                    math.degrees(az), math.degrees(el)))
            bits = ["{:.0f} fps".format(h["fps"])]
            if h["state"] != "online":
                bits.append(h["state"].upper())
            if still:
                bits.append("bias trim")
            if h["bad"]:
                bits.append("{} bad pkts".format(h["bad"]))
            self.canvas.itemconfigure(self.statusbar, text="    ".join(bits))

        self.root.after(40, self.update)

    def quit(self):
        if self.log:
            try:
                self.log[0].flush()
                self.log[0].close()
            except Exception:
                pass
            self.log = None
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
    ap.add_argument("--sim", action="store_true",
                    help="synthetic probe: run the whole UI with no hardware")
    ap.add_argument("--log", metavar="FILE.csv",
                    help="record time, az, el, quaternion for drift analysis")
    args = ap.parse_args()

    link = (SimLink() if args.sim else ProbeLink(args.port, args.baud)).start()
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
