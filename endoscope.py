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
APP_VER = "3.3"

TYPE_IMU = 1
TYPE_FRAME = 2
TYPE_RAW = 3        # one uncompressed RGB565 frame, for the DIAG grid

# Corrupted-length guard, per packet type. A corrupted length below the cap
# makes the parser sit waiting for bytes that never come, so the caps are as
# tight as the real traffic allows: IMU JSON is ~200 bytes, frames ~10-40 KB.
MAX_LEN = {TYPE_IMU: 4 * 1024, TYPE_FRAME: 256 * 1024,
           TYPE_RAW: 256 * 1024}

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


def grey_world(bgr, prev_gains, clamp=2.6, smooth=0.15):
    """
    Correct the GC0308's colour cast by equalising the channel means.

    Measured on a real frame from this probe: red 148, green 153, blue 67,
    with red/green correlated at 0.97 and blue still correlated at 0.65 with
    the image structure. All three channels therefore carry the correct
    picture and blue is simply starved -- a gain fault, not a pixel-format
    fault. That is why none of the byte-order or YUV modes ever helped: they
    were attacking a problem that was not there.

    Gains are clamped so a genuinely single-coloured scene (staring into a
    red pipe) cannot be bleached to grey, and smoothed over time so the
    picture does not pulse as the view changes.
    """
    small = bgr[::8, ::8].reshape(-1, 3).astype(np.float32)
    means = small.mean(0)
    if float(means.mean()) < 4.0:            # near-black frame: leave it alone
        return bgr, prev_gains
    gains = np.clip(means.mean() / np.maximum(means, 1e-3), 1.0 / clamp, clamp)
    if prev_gains is not None:
        gains = prev_gains + smooth * (gains - prev_gains)
    out = np.clip(bgr.astype(np.float32) * gains, 0, 255).astype(np.uint8)
    return out, gains


def aim_angles(q_now, q_ref, axis_idx, el_sign=1.0):
    """
    Return (azimuth, elevation) in radians.

    Elevation is measured from gravity, so it is absolute and drift-free.
    Azimuth is measured from the heading captured at zero time, positive to
    the operator's right, because heading has no absolute reference available
    in a steel environment.

    el_sign is the saved up/down polarity, +1 or -1. It exists because the
    lens axis and its antipode produce IDENTICAL azimuth -- negating the
    forward vector negates both the reference and the current heading, so the
    cross and dot products that define azimuth are unchanged -- while
    elevation flips. Up/down reversed with left/right still correct is
    therefore always this one bit, and nothing else. Applying it here, last,
    means neither the axis detector nor a re-zero can silently undo the
    operator's correction.
    """
    fwd = AXES[axis_idx][1]
    v = v_norm(q_rotate(q_now, fwd))
    el = math.asin(clamp(v[2])) * el_sign

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


# ---------------------------------------------------- lens-axis detection

class AxisDetector:
    """
    Identify which body axis the lens looks along, from one gesture.

    Two phases. ARMING: wait until the probe is held still for about half a
    second, then take THAT attitude as the baseline. Pressing ZERO involves
    hand motion, and v3.1 armed instantly with the zero attitude as baseline:
    a natural hand drop right after the press read as a 20-degree "tilt" and
    locked the BACKWARD axis -- which shows up as up/down inverted while
    left/right stays correct (field-observed 2026-09-03). Now nothing counts
    until the hand settles, and the UI tells the operator when to move.

    DETECTION: from the still baseline, when the operator TILTS THE LENS UP,
    the true forward axis -- and only it -- gains altitude. The first
    candidate to rise cleanly past TILT (winner at least MARGIN times the
    runner-up, held HOLD updates) is the lens axis. Because the lens was
    level at zero, the near-vertical body axes are never candidates.

    Rebasing also repairs the drop-then-lift sequence by itself: a probe that
    sagged 25 degrees before settling simply gets a lower baseline, and the
    lift back up still raises the true axis most.

    Remaining limit, set by geometry: from stillness, a deliberate tilt DOWN
    raises the backward axis with the same signature as a tilt up raises the
    forward one, and a deliberate pure twist raises a perpendicular axis.
    Attitude alone cannot tell those apart from the asked-for gesture, so the
    instruction ("tilt UP, don't twist") carries that last step, the
    right-turn check exposes a wrong lock within seconds, and RE-ZERO
    restarts detection. Mixed twist+tilt is rejected by the margin rule.
    """

    TILT = 0.34        # sin(20 deg) of rise before a lock is considered
    MARGIN = 2.0       # winner must lead the runner-up by this factor
    HOLD = 8           # consecutive qualifying updates (~0.3 s at 25 Hz)
    STILL_N = 12       # samples the pose must hold before arming (~0.5 s)
    STILL_EPS = 0.06   # max altitude wander of any candidate while "still"

    def __init__(self, q_ref):
        zs = [q_rotate(q_ref, a[1])[2] for a in AXES]
        # The lens was level at zero, so it must be one of the axes that
        # started horizontal; the near-vertical pair can never be it.
        self.cands = [i for i, b in enumerate(zs) if abs(b) < 0.5]
        self.armed = False
        self.base = None
        self.win = []
        self.hits = 0
        self.last = None

    def _zs(self, q):
        return [q_rotate(q, AXES[i][1])[2] for i in self.cands]

    def step(self, q):
        """Feed one attitude sample; returns a locked axis index or None."""
        if not self.cands:
            return None
        zs = self._zs(q)

        if not self.armed:
            self.win.append(zs)
            if len(self.win) > self.STILL_N:
                del self.win[0]
            if len(self.win) == self.STILL_N:
                spread = max(max(c) - min(c) for c in zip(*self.win))
                if spread < self.STILL_EPS:
                    self.armed = True
                    self.base = zs          # rebase on the settled attitude
            return None

        best_i, best, second = -1, -2.0, -2.0
        for k, i in enumerate(self.cands):
            s = zs[k] - self.base[k]
            if s > best:
                best_i, second, best = i, best, s
            elif s > second:
                second = s
        clean = best > self.TILT and (second <= 0.0
                                      or best > self.MARGIN * second)
        if clean and best_i == self.last:
            self.hits += 1
            if self.hits >= self.HOLD:
                return best_i
        else:
            self.hits = 1 if clean else 0
            self.last = best_i if clean else None
        return None


# ---------------------------------------------------- colour diagnostics

def photo_score(bgr):
    """
    How much does this look like a photograph rather than misread bytes?

    Two properties survive in a correctly decoded image and collapse in an
    incorrectly decoded one:

      channel agreement  the three planes describe the same scene, so they
                         correlate strongly. Reading the wrong bits puts
                         different information in each plane.
      spatial smoothness neighbouring pixels are similar. A wrong byte offset
                         alternates between two unrelated values every pixel,
                         which shows up as huge horizontal differences.

    Scored on the horizontal axis specifically: byte-order faults in a packed
    format misalign along the scan line, not down it, so a correct image and a
    misread one differ far more left-to-right than top-to-bottom.
    """
    a = bgr.astype(np.float32)
    B, G, R = a[..., 0].ravel(), a[..., 1].ravel(), a[..., 2].ravel()

    def cor(x, y):
        x = x - x.mean(); y = y - y.mean()
        d = math.sqrt(float((x * x).sum())) * math.sqrt(float((y * y).sum()))
        return float((x * y).sum() / d) if d > 1e-6 else 0.0

    agree = (cor(R, G) + cor(G, B) + cor(R, B)) / 3.0

    grey = a.mean(2)
    dh = np.abs(np.diff(grey, axis=1)).mean()
    dv = np.abs(np.diff(grey, axis=0)).mean()
    # Ratio, not absolute: a flat wall and a busy workshop then score alike.
    smooth = 1.0 / (1.0 + dh / max(dv, 1e-3))

    return 0.6 * agree + 0.4 * smooth, agree, dh, dv


def interpretations(raw, width, height):
    """Every plausible reading of one raw sensor frame, as BGR images."""
    if len(raw) < width * height * 2:
        return []
    buf = np.frombuffer(raw, dtype=np.uint8,
                        count=width * height * 2).reshape(height, width, 2)

    def rgb565(le):
        w16 = ((buf[:, :, 0].astype(np.uint16) | (buf[:, :, 1].astype(np.uint16) << 8))
               if le else
               (buf[:, :, 1].astype(np.uint16) | (buf[:, :, 0].astype(np.uint16) << 8)))
        r = (((w16 >> 11) & 0x1F).astype(np.uint16) * 527 + 23) >> 6
        g = (((w16 >> 5) & 0x3F).astype(np.uint16) * 259 + 33) >> 6
        b = ((w16 & 0x1F).astype(np.uint16) * 527 + 23) >> 6
        return np.dstack([b, g, r]).astype(np.uint8)        # BGR for cv2

    return [("RGB565 hi-first = mode 0", 0, rgb565(False)),
            ("RGB565 lo-first = mode 1", 1, rgb565(True)),
            ("YUV YUYV = mode 2", 2, cv2.cvtColor(buf, cv2.COLOR_YUV2BGR_YUY2)),
            ("YUV YVYU", None, cv2.cvtColor(buf, cv2.COLOR_YUV2BGR_YVYU)),
            ("YUV UYVY", None, cv2.cvtColor(buf, cv2.COLOR_YUV2BGR_UYVY))]


def build_diag_grid(raw, width, height, save_dir=None):
    """
    Score every interpretation and lay them out with the numbers visible, so
    the choice stops depending on anyone squinting at a screen. Returns
    (grid image, best mode or None, report lines).

    Also dumps the raw bytes: if the scores still disagree with what the eye
    sees, that file settles it offline in one step instead of another round of
    photograph-and-guess.
    """
    items = interpretations(raw, width, height)
    if not items:
        return None, None, []

    scored = []
    for name, mode, img in items:
        sc, agree, dh, dv = photo_score(img)
        scored.append((sc, agree, dh, dv, name, mode, img))

    best = max(scored, key=lambda t: t[0])
    report = ["raw frame: {} bytes, {}x{}".format(len(raw), width, height)]
    for sc, agree, dh, dv, name, mode, _ in scored:
        report.append("  {:26s} score {:.3f}  agree {:+.3f}  dH {:5.1f}  dV {:5.1f}{}"
                      .format(name, sc, agree, dh, dv,
                              "   <-- best" if name == best[4] else ""))

    if save_dir:
        try:
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, "endoscope_raw.bin"), "wb") as f:
                f.write(raw)
            report.append("  raw bytes written to "
                          + os.path.join(save_dir, "endoscope_raw.bin"))
        except Exception as e:
            report.append("  (could not save raw: {})".format(e))

    label_h = 30
    th, tw = height + label_h, width
    grid = np.full((th * 2, tw * 3, 3), 16, np.uint8)
    for i, (sc, agree, dh, dv, name, mode, img) in enumerate(scored):
        r, c = divmod(i, 3)
        y, x = r * th, c * tw
        grid[y + label_h:y + th, x:x + tw] = img
        win = name == best[4]
        cv2.putText(grid, "{} {:.2f}{}".format(name, sc, "  BEST" if win else ""),
                    (x + 8, y + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (120, 255, 160) if win else (235, 240, 245), 1)
        if win:
            cv2.rectangle(grid, (x + 1, y + 1), (x + tw - 2, y + th - 2),
                          (120, 255, 160), 2)
    cv2.putText(grid, "auto-scored - no photo needed",
                (tw * 2 + 8, th + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (140, 220, 160), 1)
    return grid, best[5], report


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
        self.fw_ver = None              # (major, rev) from the ready line
        self.colour_mode = None         # probe's current colour mode, if told
        self.calib_ok = None            # gyro calibration verdict, if told
        self.raw = None                 # last raw frame for the DIAG grid
        self.raw_seq = 0
        self.calibrating = False
        self.camera_failed = False
        self.generation = 0
        self.imu_time = 0.0
        self.frame_time = 0.0
        self.status = []
        self.bad_packets = 0
        self._fps = []
        self._wlock = threading.Lock()
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
                self.ser = serial.Serial(port, self.baud, timeout=0.5,
                                         write_timeout=0.3)
            except Exception:
                self._set_state("offline")
                time.sleep(1.5)
                continue
            self.port = port
            self.buf.clear()
            with self.lock:
                self.generation += 1
                self.fw_ver = None      # fresh boot: wait for its ready line
                self.colour_mode = None
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

        elif ptype == TYPE_RAW:
            with self.lock:
                self.raw = bytes(payload)
                self.raw_seq += 1

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

    def send_byte(self, b):
        """
        One command byte to the probe (v3.2+ reads them; older firmware just
        leaves them in its RX buffer, hence the write timeout). Returns
        whether the byte was handed to the driver.
        """
        ser = self.ser
        if ser is None:
            return False
        try:
            with self._wlock:
                ser.write(bytes([b]))
            return True
        except Exception:
            return False

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
                if st == "ready":
                    try:
                        v = data.get("v")
                        self.fw_ver = ((int(v), int(data.get("r", 0)))
                                       if v is not None else (0, 0))
                    except (TypeError, ValueError):
                        self.fw_ver = (0, 0)
                    cm = data.get("cm")
                    self.colour_mode = (int(cm) if isinstance(cm, (int, float))
                                        else None)
                    cal = data.get("calib")
                    self.calib_ok = (bool(cal) if cal is not None else None)
            elif st == "colour_mode":
                try:
                    self.colour_mode = int(data.get("mode", 0))
                except (TypeError, ValueError):
                    pass

    # ---- what the UI reads

    def snapshot(self):
        with self.lock:
            return self.frame, self.frame_seq, self.quat, self.still

    def take_raw(self):
        with self.lock:
            return self.raw, self.raw_seq

    def health(self):
        now = time.monotonic()
        with self.lock:
            self._fps = [t for t in self._fps if now - t < 2.0]
            return {
                "state": self.state,
                "fw": self.fw,
                "fw_ver": self.fw_ver,
                "colour_mode": self.colour_mode,
                "calib_ok": self.calib_ok,
                "raw_seq": self.raw_seq,
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
        self.fw_ver = (3, 3)
        self.colour_mode = 1
        self.calib_ok = True
        self.raw = None
        self.raw_seq = 0
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
        # Scripted CHECK sequence, mirroring a real operator so the axis
        # auto-detection runs end-to-end in --sim:
        #   t 2.6..4.2   hold still, slightly nose-up (detector arms here)
        #   t 4.2..7.2   deliberate tilt-up to 50 deg and back
        yaw_t, pitch = t, 30.0 * math.sin(2 * math.pi * t / 9.0)
        if 2.6 < t < 7.2:
            yaw_t, pitch = 2.6, 10.0
            if t > 4.2:
                g = min((t - 4.2) / 1.2, 1.0, (7.2 - t) / 1.0)
                pitch = 10.0 + 40.0 * max(g, 0.0)
        yaw = math.radians(50.0 * math.sin(2 * math.pi * yaw_t / 24.0))
        pitch = math.radians(pitch)
        qz = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        qy = (math.cos(pitch / 2), 0.0, -math.sin(pitch / 2), 0.0)
        return q_mul(qz, qy)            # lens along +X, nose-up positive

    def send_byte(self, b):
        if b in (ord("c"), ord("C")):
            with self.lock:
                self.colour_mode = (self.colour_mode + 1) % 3
        elif b in (ord("0"), ord("1"), ord("2")):
            with self.lock:
                self.colour_mode = b - ord("0")
        elif b in (ord("r"), ord("R")):
            with self.lock:
                f = self.frame
            if f is not None:                    # BGR -> RGB565 little-endian
                r = (f[:, :, 2] >> 3).astype(np.uint16)
                g = (f[:, :, 1] >> 2).astype(np.uint16)
                bl = (f[:, :, 0] >> 3).astype(np.uint16)
                w16 = (r << 11) | (g << 5) | bl
                with self.lock:
                    self.raw = w16.astype("<u2").tobytes()
                    self.raw_seq += 1
        return True

    def take_raw(self):
        with self.lock:
            return self.raw, self.raw_seq

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
                    "fw_ver": self.fw_ver, "colour_mode": self.colour_mode,
                    "calib_ok": self.calib_ok, "raw_seq": self.raw_seq,
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
        self.axis_det = None            # armed by capture_zero
        self.axis_auto = False          # True: the tilt-up gesture may relock
        self.axis_btn_text = None       # CHECK's AXIS button label item
        self.check_hint = None          # CHECK's first guidance line
        self._hint_state = ""           # wait / armed / locked
        self._last_cm = None            # last colour mode we toasted about
        self.rot180 = bool(self.cfg.get("rot180", True))
        # Up/down polarity, set once with FLIP U/D and never touched
        # by zeroing or by the axis detector.
        # Default is INVERTED: on this build of the probe the lens axis the
        # detector locks is the antipode of the true one, so a lift read as
        # a drop until FLIP U/D was pressed every session. Baked in; the
        # button stays so a differently-mounted probe can undo it.
        self.el_sign = 1.0 if self.cfg.get("el_sign", -1) > 0 else -1.0
        self.awb = bool(self.cfg.get("awb", True))
        self.swap_rb = bool(self.cfg.get("swap_rb", False))
        self._awb_gains = None
        self.diag_photo = None          # sticky DIAG grid, shown over video
        self._raw_seen = 0
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
            self.root.attributes("-topmost", True)
            self.root.overrideredirect(True)
            self.root.configure(cursor="none")
            self.root.update_idletasks()
            self.W = self.root.winfo_screenwidth()
            self.H = self.root.winfo_screenheight()
            # A window manager, a notification or a stray key can drop a
            # window out of fullscreen. On a fixed-function instrument that
            # must never happen, so the state is re-asserted rather than
            # merely requested once.
            self._keep_fullscreen()

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
                json.dump({"axis": self.axis_idx,
                           "rot180": self.rot180,
                           "el_sign": self.el_sign,
                           "awb": self.awb,
                           "swap_rb": self.swap_rb}, f)
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
        self.axis_det = AxisDetector(q)
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
        z.append(c.create_text(W - max(10, H // 46), H - max(10, H // 46),
                               anchor="se", text="app v" + APP_VER,
                               fill=MUTED, font=self.f(small)))

    def do_zero(self):
        if not self.capture_zero():
            return
        self.axis_auto = True           # the tilt-up move may (re)set the axis
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
        self._hint_state = "wait"
        self.check_hint = c.create_text(
            W / 2, y,
            text="HOLD THE PROBE STILL for a moment \u2026",
            fill="#b0bec5", font=self.f(tiny, True))
        z.append(self.check_hint)
        y += tiny * 2 + H * 0.006
        z.append(c.create_text(
            W / 2, y,
            text="Then turn right \u2014 arrow must swing right.   "
                 "Up/down reversed? Press FLIP U/D.",
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
                  ("FLIP U/D", "#6d4c41", self.flip_axis),
                  ("RE-ZERO", "#1565c0", self.do_zero),
                  ("START", "#2e7d32", self.start_run)]
        widths = [self.text_w(t, bs, True) + 52 for t, _, _ in labels]
        total = sum(widths) + gap * (len(widths) - 1)
        x = W / 2 - total / 2
        by = H - pad - (bs * 2 + 30) / 2
        for (label, col, cb), w in zip(labels, widths):
            _, t, _, _ = self.button(x + w / 2, by, label, col, cb, bs,
                                     store=z)
            if cb == self.cycle_axis:
                self.axis_btn_text = t
            x += w + gap

    def flip_axis(self):
        """
        Invert the up/down reading, and keep it inverted.

        v3.3 flipped the lens axis to its antipode instead. That is the same
        maths -- polarity only moves elevation -- but it lived in the axis
        field, so the next ZERO let the tilt-up detector relock a polarity of
        its own choosing and the correction vanished. The operator saw it come
        back wrong every session. The sign now lives in its own saved field
        that only this button writes.
        """
        self.el_sign = -self.el_sign
        self.save_cfg()
        self.peak_az = 0.0
        self.toast("UP/DOWN " + ("NORMAL" if self.el_sign > 0 else "INVERTED")
                   + " \u2713", OK, ms=2000)

    def cycle_axis(self):
        # A manual choice wins over the detector until the next ZERO/RE-ZERO,
        # otherwise the next tilt would immediately override it.
        self.axis_auto = False
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
        self.button(pad, H - pad - int(H * 0.055), "COLOR", "#455a64",
                    self.cycle_colour, max(10, int(H / 38)),
                    store=self.hud, anchor="sw")
        self.button(pad, H - pad - int(H * 0.055) - int(H * 0.085), "DIAG",
                    "#4e5d3a", self.toggle_diag, max(10, int(H / 38)),
                    store=self.hud, anchor="sw")

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

    def cycle_colour(self):
        """
        Ask the probe for its next colour mode (0 raw / 1 byte-swap / 2 YUV).
        A scrambled picture is fixed by pressing this until it looks natural:
        the failure is in how the sensor's bytes are interpreted before JPEG,
        so it can only be fixed probe-side, and cycling live beats reflashing
        once per guess. The probe acknowledges with a status packet, which
        pops the COLOUR MODE toast below.
        """
        h = self.link.health()
        fv = h.get("fw_ver")
        if fv is None or fv < (3, 2) or h.get("colour_mode") is None:
            self.toast("PROBE FIRMWARE HAS NO COLOUR SWITCH \u2014 FLASH v3.2",
                       WARN, ms=2600)
            return
        if not self.link.send_byte(ord("c")):
            self.toast("PROBE NOT CONNECTED", WARN)

    def toggle_diag(self):
        """
        Ask the probe for one RAW frame and freeze an interpretation grid on
        screen (photograph it); press again to go back to live video. This
        exists because cycling three colour modes blind can still leave "all
        wrong", and the grid settles in one shot what the sensor really sends.
        """
        if self.diag_photo is not None:
            self.diag_photo = None
            return
        h = self.link.health()
        fv = h.get("fw_ver")
        if fv is None or fv < (3, 3):
            self.toast("PROBE FIRMWARE HAS NO DIAG \u2014 FLASH v3.3",
                       WARN, ms=2600)
            return
        if self.link.send_byte(ord("r")):
            self.toast("DIAG: requesting raw frame \u2026", DIM, ms=1200)
        else:
            self.toast("PROBE NOT CONNECTED", WARN)

    def zero_inplace(self):
        """
        Re-zero without leaving the video. Heading drifts; the fix is one tap:
        aim forward, press ZERO, keep working. Going back through the check
        screen mid-inspection would only make the honest mitigation annoying.
        """
        if self.capture_zero():
            self.axis_auto = False      # never switch the axis mid-inspection
            self.toast("ZEROED \u2713", OK)

    def _keep_fullscreen(self):
        try:
            if not self.root.attributes("-fullscreen"):
                self.root.attributes("-fullscreen", True)
            self.root.attributes("-topmost", True)
        except Exception:
            return
        self.root.after(2000, self._keep_fullscreen)

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
        elif k == "c" and self.stage == self.STAGE_RUN:
            self.cycle_colour()
        elif k == "f" and self.stage == self.STAGE_CHECK:
            self.flip_axis()
        elif k == "d" and self.stage == self.STAGE_RUN:
            self.toggle_diag()
        elif k == "b" and self.stage == self.STAGE_RUN:
            self.swap_rb = not self.swap_rb
            self._awb_gains = None
            self.save_cfg()
            self.last_seq = -1
            self.toast("RED/BLUE SWAP: " + ("ON" if self.swap_rb else "OFF"),
                       OK, ms=1400)
        elif k == "w" and self.stage == self.STAGE_RUN:
            # Escape hatch: if a scene really is one colour, AWB will bleach
            # it, and the operator needs to be able to see the sensor plain.
            self.awb = not self.awb
            self._awb_gains = None
            self.save_cfg()
            self.last_seq = -1
            self.toast("AUTO WHITE BALANCE: " + ("ON" if self.awb else "OFF"),
                       OK, ms=1400)
        elif k == "v" and self.stage == self.STAGE_RUN:
            self.rot180 = not self.rot180
            self.save_cfg()
            self.last_seq = -1          # redraw the current frame flipped
            self.toast("ROTATE 180: " + ("ON" if self.rot180 else "OFF"),
                       OK, ms=1200)

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
        if h.get("calib_ok") is False:
            return (s + " \u00b7 GYRO CALIB FAILED \u2014 reboot the probe "
                    "and keep it still", WARN)
        fv = h.get("fw_ver")
        if fv == (0, 0):
            # It answered "ready" without a version: pre-v3 firmware.
            return s + " \u00b7 fw OLD \u2014 FLASH v3.2", WARN
        if fv is not None:
            s += " \u00b7 fw {}.{}".format(fv[0], fv[1])
            if fv < (3, 2):
                return s + " \u2014 flash v3.2 for the colour switch", WARN
            if h.get("colour_mode") is not None:
                s += " \u00b7 colour {}".format(h["colour_mode"])
            fv = h.get("fw_ver")
            if fv is not None and fv < (3, 4):
                # The commonest cause of "still garbled" is the Pi being
                # updated while the probe was not. Say so outright instead of
                # letting it look like a colour bug.
                s += " \u00b7 OLD FIRMWARE - RE-FLASH"
        return s, OK

    def update(self):
        frame, seq, q, still = self.link.snapshot()
        h = self.link.health()
        az, el = aim_angles(q, self.q_ref, self.axis_idx, self.el_sign)
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
            if self.axis_auto and self.axis_det is not None:
                hit = self.axis_det.step(q)
                if hit is None:
                    want = "armed" if self.axis_det.armed else "wait"
                    if want != self._hint_state and self.check_hint:
                        self._hint_state = want
                        self.canvas.itemconfigure(
                            self.check_hint,
                            text=("NOW TILT THE LENS UP \u2014 this finds the "
                                  "lens axis. Tilt, don't twist."
                                  if want == "armed" else
                                  "HOLD THE PROBE STILL for a moment \u2026"))
                else:
                    self.axis_auto = False
                    self._hint_state = "locked"
                    if hit != self.axis_idx:
                        self.axis_idx = hit
                        self.save_cfg()
                        # Heading is defined per-axis, so recompute this
                        # frame's angles and restart the drift peak.
                        az, el = aim_angles(q, self.q_ref, self.axis_idx, self.el_sign)
                        self.peak_az = 0.0
                        self.toast("LENS AXIS AUTO-SET: "
                                   + AXES[hit][0] + " \u2713", OK, ms=2400)
                    else:
                        self.toast("LENS AXIS CONFIRMED: "
                                   + AXES[hit][0] + " \u2713", OK, ms=2400)
                    if self.axis_btn_text:
                        self.canvas.itemconfigure(
                            self.axis_btn_text,
                            text="AXIS " + AXES[self.axis_idx][0])
                    if self.check_hint:
                        self.canvas.itemconfigure(
                            self.check_hint,
                            text="Axis " + AXES[self.axis_idx][0]
                                 + " locked \u2713   Tilt up \u2014 arrow "
                                   "grows.   Turn right \u2014 arrow right.")
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
            cm = h.get("colour_mode")
            if cm is not None and cm != self._last_cm:
                prev, self._last_cm = self._last_cm, cm
                if prev is not None:      # not the first report after boot
                    names = {0: "RAW", 1: "BYTE-SWAP", 2: "YUV"}
                    self.toast("COLOUR MODE {} \u2014 {}".format(
                        cm, names.get(cm, "?")), OK, ms=1600)
            if h.get("raw_seq", 0) != self._raw_seen:
                raw, self._raw_seen = self.link.take_raw()
                grid, best, report = (build_diag_grid(
                    raw, 320, 240, save_dir=os.path.expanduser("~"))
                    if raw else (None, None, []))
                if grid is not None:
                    self.diag_photo = grid
                    for line in report:
                        print(line)
                    if best is not None and best != self.link.health().get(
                            "colour_mode"):
                        # The measurement decided; apply it rather than
                        # asking anyone to interpret the picture.
                        self.link.send_byte(ord('0') + best)
                        self.toast("COLOUR MODE {} chosen automatically"
                                   .format(best), OK, ms=3000)
                    else:
                        self.toast("DIAG scored \u2014 press DIAG to leave",
                                   OK, ms=3000)

            show = self.diag_photo if self.diag_photo is not None else frame
            fresh = (self.diag_photo is not None) or (frame is not None
                                                      and seq != self.last_seq)
            if show is not None and fresh:
                self.last_seq = seq
                if self.diag_photo is None and self.rot180:
                    # The camera module sits upside-down in the case, so the
                    # picture is delivered rotated; flip it for the operator.
                    # DIAG stays unrotated: it must show the sensor's truth.
                    show = cv2.rotate(show, cv2.ROTATE_180)
                fh, fw = show.shape[:2]
                scale = min(self.W / fw, self.H / fh)
                small = cv2.resize(show, (int(fw * scale), int(fh * scale)),
                                   interpolation=cv2.INTER_LINEAR)
                if self.swap_rb and self.diag_photo is None:
                    # A red/blue transposition survives JPEG intact, so unlike
                    # a bit-order fault it can still be undone here.
                    small = small[:, :, ::-1]
                if self.awb and self.diag_photo is None:
                    small, self._awb_gains = grey_world(small, self._awb_gains)
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
                self.canvas.itemconfigure(self.video_item, image=self.photo)
                self.canvas.tag_lower(self.video_item)
                for i in self.hud:
                    self.canvas.tag_raise(i)

            stale = h["frame_age"] > 2.5 and self.diag_photo is None
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
            if self.rot180:
                bits.append("ROT180")
            if self.awb:
                bits.append("AWB")
            if self.diag_photo is not None:
                bits.append("DIAG")
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
