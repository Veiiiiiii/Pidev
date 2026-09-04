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
    python3 endoscope.py --official         # stock M5 firmware: UVC + IMU WiFi
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
import base64
import csv
import glob
import hashlib
import json
import math
import os
import socket
import struct
import sys
import threading
import time
from urllib.parse import urlsplit

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
APP_VER = "5.0"
CONFIG_REV = 6

TYPE_IMU = 1
TYPE_FRAME = 2
TYPE_RAW = 3        # uncompressed RGB565: DIAG one-shot, or the raw stream
RAW_STREAM_W = 160  # the probe halves the frame for the raw stream
RAW_STREAM_H = 120

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
# Flip buttons: filled accent when engaged, dark slate when off, so their
# state reads from across a workshop without squinting at the label.
FLIP_ON = "#26a69a"
FLIP_OFF = "#37474f"
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



def analyze_bars(raw, width, height):
    """
    Judge the sensor's internal colour bars. Each bar should be a corner of
    the RGB cube (every channel near 0 or near full). That is independent of
    bar order and of any white balance, because the pattern is generated
    inside the chip: it tests only the parallel data path and the unpack.
    Returns (image, verdict_text, passed).
    """
    if len(raw) < width * height * 2:
        return None, "raw frame too short", False
    buf = np.frombuffer(raw, dtype=np.uint8, count=width * height * 2).reshape(height, width, 2)
    w16 = (buf[:, :, 0].astype(np.uint16) << 8) | buf[:, :, 1].astype(np.uint16)
    r = (((w16 >> 11) & 0x1F) << 3).astype(np.uint8)
    g = (((w16 >> 5) & 0x3F) << 2).astype(np.uint8)
    b = ((w16 & 0x1F) << 3).astype(np.uint8)
    img = np.dstack([b, g, r])

    def strips(vertical):
        out = []
        for i in range(8):
            if vertical:
                s = img[int(height * 0.2):int(height * 0.8),
                        int(width * i / 8) + 4:int(width * (i + 1) / 8) - 4]
            else:
                s = img[int(height * i / 8) + 3:int(height * (i + 1) / 8) - 3,
                        int(width * 0.2):int(width * 0.8)]
            out.append(s.reshape(-1, 3).mean(axis=0))
        return out

    def score(means):
        ok = 0
        for m in means:
            if all(v < 70 or v > 185 for v in m):
                ok += 1
        distinct = len({tuple(int(v > 128) for v in m) for m in means})
        return ok, distinct

    v_means, h_means = strips(True), strips(False)
    v_ok, v_d = score(v_means)
    h_ok, h_d = score(h_means)
    means, ok, dist, orient = ((v_means, v_ok, v_d, "vertical")
                               if (v_ok, v_d) >= (h_ok, h_d) else
                               (h_means, h_ok, h_d, "horizontal"))
    passed = ok >= 6 and dist >= 6

    out = np.full((height * 2 + 150, max(width * 2, 900), 3), 16, np.uint8)
    out[0:height * 2, 0:width * 2] = cv2.resize(img, (width * 2, height * 2),
                                                interpolation=cv2.INTER_NEAREST)
    sw = (width * 2) // 8
    for i, m in enumerate(means):
        x0 = i * sw
        out[height * 2 + 10:height * 2 + 60, x0 + 4:x0 + sw - 4] = m.astype(np.uint8)
        cv2.putText(out, "R%3d G%3d B%3d" % (m[2], m[1], m[0]), (x0 + 4, height * 2 + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 235, 240), 1)
    verdict = ("SENSOR BARS PASS  {}/8 clean, {} distinct ({}) -> data path OK; "
               "the cast is colour processing".format(ok, dist, orient) if passed else
               "SENSOR BARS FAIL  {}/8 clean, {} distinct ({}) -> data path suspect "
               "(pins / PCLK / bit order)".format(ok, dist, orient))
    cv2.putText(out, verdict, (8, height * 2 + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (120, 235, 140) if passed else (110, 110, 250), 2)
    return out, verdict, passed


# ------------------------------------------------------------- serial link

def find_port(preferred=None):
    if preferred:
        return preferred if os.path.exists(preferred) else None
    for pattern in ("/dev/ttyACM*", "/dev/ttyUSB*"):
        ports = sorted(glob.glob(pattern))
        if ports:
            return ports[0]
    return None


def find_video(preferred="auto"):
    """Resolve a V4L2 node, preferring the AtomS3R-CAM by its kernel name."""
    if preferred not in (None, "", "auto"):
        return int(preferred) if str(preferred).isdigit() else preferred

    nodes = sorted(glob.glob("/dev/video*"))
    if not nodes:
        return None

    ranked = []
    for node in nodes:
        name = ""
        try:
            base = os.path.basename(node)
            with open("/sys/class/video4linux/{}/name".format(base),
                      encoding="utf-8") as f:
                name = f.read().strip().lower()
        except OSError:
            pass
        score = 0
        if "atoms3r" in name or "atom s3r" in name:
            score += 100
        if "uvc" in name or "camera" in name:
            score += 10
        ranked.append((score, node))
    return max(ranked, key=lambda item: (item[0], -int(item[1][10:])))[1]


class V4L2Source:
    """
    Video from a standard UVC camera, read by V4L2 through OpenCV.

    This exists because the custom video path should never have been built.
    The AtomS3R-CAM ships as a UVC device: plug it in and the Pi gives you
    /dev/video0 with correct colour, decoded by the kernel and OpenCV. Flashing
    custom firmware to get the IMU took that away, and everything that followed
    -- a serial protocol, JPEG encoding on an ESP32, RGB565 unpacking, white
    balance, black point, gamma -- was rebuilding what UVC had provided for
    free. Every colour fault in this project lives in that rebuilt code.

    With a UVC camera supplying the picture and the probe supplying only
    attitude, none of that code runs at all.
    """

    def __init__(self, index="auto", width=640, height=480):
        self.index = index
        self.want = (width, height)
        self.cap = None
        self.lock = threading.Lock()
        self.frame = None
        self.frame_seq = 0
        self.frame_time = 0.0
        self.running = False
        self.error = None
        self._fps = []

    def start(self):
        src = find_video(self.index)
        if src is None:
            raise RuntimeError("no /dev/video device found; flash the official "
                               "AtomS3R-CAM UVC firmware and reconnect USB")
        self.index = src
        self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError("cannot open camera {}. Try: ls /dev/video*"
                               .format(self.index))
        # MJPG first: at 720p a YUYV stream will not fit down USB 2.0 at a
        # usable frame rate, and most UVC cameras offer both.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.want[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.want[1])
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, _ = self.cap.read()
        if not ok:
            raise RuntimeError("camera {} opened but returns no frames"
                               .format(self.index))
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()
        return self

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            now = time.monotonic()
            with self.lock:
                self.frame = frame
                self.frame_seq += 1
                self.frame_time = now
                self._fps.append(now)

    def fps(self):
        now = time.monotonic()
        with self.lock:
            self._fps = [t for t in self._fps if now - t < 2.0]
            return len(self._fps) / 2.0

    def snapshot(self):
        with self.lock:
            return self.frame, self.frame_seq, self.frame_time

    def stop(self):
        self.running = False
        time.sleep(0.05)
        if self.cap:
            self.cap.release()


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

    #: Millisecond gap in the probe clock that counts as lost integration
    #: time rather than ordinary jitter (50 Hz nominal = 20 ms per sample).
    GAP_MS = 120

    def __init__(self, port=None, baud=115200):
        self.want_port = port
        self.baud = baud
        self.port = None
        self.ser = None
        self.buf = bytearray()
        self.lock = threading.Lock()
        # v6 link-quality counters, reported on the check screen and in the
        # status line. imu_gap_ms is the total probe-clock time for which no
        # attitude sample arrived; that is heading error already banked.
        self.dev_time = None
        self.imu_gaps = 0
        self.imu_gap_ms = 0.0
        self.worst_gap_ms = 0.0

        self.frame = None
        self.frame_seq = 0
        self.quat = (1.0, 0.0, 0.0, 0.0)
        self.still = False

        self.state = "searching"        # searching / connecting / online / offline
        self.fw = ""                    # last firmware status string
        self.fw_ver = None              # (major, rev) from the ready line
        self.colour_mode = None         # probe's current colour mode, if told
        self.sensor_preset = None       # (n, name) from the probe, if told
        self.sensor_regs = ""           # register dump from the last preset ack
        self.sensor_bars = None
        self.regs_dict = {}             # parsed {reg: val} from the last dump
        self.regs_seq = 0
        self.defaults = None            # first dump of this boot = driver defaults
        self.reg_ack = None             # (seq, reg, val, readback, ok)
        self._ack_n = 0
        # OpenCV's COLOR_BGR5652BGR reads the pair low byte first; this
        # sensor emits high byte first, measured by the DIAG scorer (hi-first
        # 0.69 against lo-first 0.24). So the pair is swapped before handing it
        # over. Getting this backwards is what produced the concentric rainbow
        # contours: a smooth scene read at the wrong offset makes the low
        # colour fields cycle instead of climb. Toggle live with X.
        self.raw_swap = True            # byte order for the raw stream
        self.raw_stream = False         # probe is sending unconverted bytes
        self.camera = None              # optional UVC source, overrides video
        self.test_pattern = False       # probe is sending synthetic bars
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
                self.defaults = None
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
                # "t" is the probe's own millisecond clock at the moment the
                # sample was integrated. It is the only trustworthy timeline:
                # arrival time on the Pi includes USB scheduling and Python
                # wake-up jitter. Gaps in it are the measurement that says
                # whether heading can still be believed -- attitude is an
                # INTEGRAL, so time the probe spent rotating while packets
                # were missing is error that never comes back on its own.
                dev_t = data.get("t")
                with self.lock:
                    self.quat = tuple(q)
                    self.still = bool(data.get("st"))
                    self.imu_time = time.monotonic()
                    if isinstance(dev_t, (int, float)):
                        if self.dev_time is not None:
                            gap_ms = dev_t - self.dev_time
                            # 50 Hz nominal => 20 ms. Anything past 120 ms is
                            # several lost samples, not ordinary jitter.
                            if gap_ms > 120:
                                self.imu_gaps += 1
                                self.imu_gap_ms += gap_ms
                                self.worst_gap_ms = max(self.worst_gap_ms,
                                                        gap_ms)
                            elif gap_ms < 0:
                                # Probe rebooted: its clock restarted, so any
                                # zero reference captured before is garbage.
                                self.imu_gaps += 1
                        self.dev_time = float(dev_t)

        elif ptype == TYPE_RAW:
            payload = bytes(payload)
            small = len(payload) == RAW_STREAM_W * RAW_STREAM_H * 2
            if small:
                # Raw stream: hand the bytes to OpenCV's own RGB565 decoder.
                # No firmware conversion was involved, so nothing upstream can
                # have misread them; if this looks wrong the sensor data is
                # wrong, which is a different fault entirely.
                buf = np.frombuffer(payload, dtype=np.uint8).reshape(
                    RAW_STREAM_H, RAW_STREAM_W, 2)
                if self.raw_swap:
                    buf = buf[:, :, ::-1].copy()
                img = cv2.cvtColor(buf, cv2.COLOR_BGR5652BGR)
                now = time.monotonic()
                with self.lock:
                    self.frame = img
                    self.frame_seq += 1
                    self.frame_time = now       # what frame_age reads
                    self._fps.append(now)
            else:
                with self.lock:
                    self.raw = payload
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

    def send_bytes(self, bs):
        """
        Command bytes to the probe (v3.2+ reads them; older firmware just
        leaves them in its RX buffer, hence the write timeout). Returns
        whether the bytes were handed to the driver.
        """
        ser = self.ser
        if ser is None:
            return False
        try:
            with self._wlock:
                ser.write(bytes(bs))
            return True
        except Exception:
            return False

    def send_byte(self, b):
        return self.send_bytes([b])

    def get_regs(self):
        with self.lock:
            return dict(self.regs_dict), (dict(self.defaults) if self.defaults else None)

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
                    if "sp" in data:
                        try:
                            self.sensor_preset = (int(data["sp"]), "")
                        except (TypeError, ValueError):
                            pass
            elif st == "test_pattern":
                self.test_pattern = bool(data.get("on"))
            elif st == "raw_stream":
                self.raw_stream = bool(data.get("on"))
            elif st == "colour_mode":
                try:
                    self.colour_mode = int(data.get("mode", 0))
                except (TypeError, ValueError):
                    pass
            elif st == "regs":
                d = {}
                for tok in str(data.get("regs", "")).split():
                    if "=" in tok:
                        a, b = tok.split("=", 1)
                        try:
                            d[int(a, 16)] = int(b, 16)
                        except ValueError:
                            pass
                if d:
                    self.regs_dict = d
                    self.regs_seq += 1
                    if self.defaults is None:
                        self.defaults = dict(d)
            elif st == "reg":
                try:
                    self._ack_n += 1
                    self.reg_ack = (self._ack_n,
                                    int(str(data.get("reg", "0")), 16),
                                    int(str(data.get("val", "0")), 16),
                                    int(str(data.get("rb", "0")), 16),
                                    bool(data.get("ok")))
                except ValueError:
                    pass
            elif st == "sensor_bars":
                self.sensor_bars = (bool(data.get("on")), bool(data.get("ok")))
            elif st == "sensor_preset":
                try:
                    # Carry the probe's proof along: did the register writes
                    # land (readback), and does the bus see the sensor (PID)?
                    nm = "{} \u00b7 {} pid {} r24 {}".format(
                        data.get("name", ""),
                        "OK" if data.get("ok") else "WRITE FAILED",
                        data.get("pid", "?"), data.get("reg24", "?"))
                    self.sensor_preset = (int(data.get("n", 0)), nm)
                    self.sensor_regs = str(data.get("regs", ""))
                    if "cm" in data:
                        self.colour_mode = int(data["cm"])
                except (TypeError, ValueError):
                    pass

    # ---- what the UI reads

    def snapshot(self):
        if self.camera is not None:
            frame, seq, _ = self.camera.snapshot()
            with self.lock:
                return frame, seq, self.quat, self.still
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
                "sensor_preset": self.sensor_preset,
                "sensor_regs": self.sensor_regs,
                "regs_seq": self.regs_seq,
                "reg_ack": self.reg_ack,
                "raw_stream": self.raw_stream,
                "test_pattern": self.test_pattern,
                "calib_ok": self.calib_ok,
                "raw_seq": self.raw_seq,
                "calibrating": self.calibrating,
                "camera_failed": self.camera_failed,
                "generation": self.generation,
                "imu_age": now - self.imu_time if self.imu_time else 1e9,
                "frame_age": ((now - self.camera.frame_time)
                              if self.camera is not None and self.camera.frame_time
                              else (now - self.frame_time if self.frame_time else 1e9)),
                "fps": (self.camera.fps() if self.camera is not None
                        else len(self._fps) / 2.0),
                "bad": self.bad_packets,
                # v6: lost integration time. Non-zero here is the honest
                # reason a heading stopped matching reality.
                "imu_gaps": self.imu_gaps,
                "imu_gap_ms": self.imu_gap_ms,
                "worst_gap_ms": self.worst_gap_ms,
                "port": self.port,
            }


class MahonyFusion:
    """The proven probe-side attitude filter, for the stock-firmware route."""

    KP = 2.0
    KI = 0.05
    STILL_GYRO_DPS = 1.5
    STILL_ACC_TOL = 0.06
    STILL_S = 0.5
    BIAS_LEARN = 0.01       # stock IMU WebSocket is 10 Hz, not 100 Hz
    BIAS_CLAMP_DPS = 3.0
    CAL_SAMPLES = 30        # three seconds at the stock 10 Hz rate

    def __init__(self):
        self.reset()

    def reset(self):
        self.q = [1.0, 0.0, 0.0, 0.0]
        self.bias = [0.0, 0.0, 0.0]
        self.cal_bias = [0.0, 0.0, 0.0]
        self.integral = [0.0, 0.0, 0.0]
        self.last_time = None
        self.still_since = None
        self.still = False
        self.calibrated = False
        self.cal_window = []

    def _seed_from_gravity(self, ax, ay, az):
        n = math.sqrt(ax * ax + ay * ay + az * az)
        if n < 0.5:
            return
        ax, ay, az = ax / n, ay / n, az / n
        pitch = math.atan2(-ax, math.sqrt(ay * ay + az * az))
        roll = math.atan2(ay, az)
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
        self.q = [cr * cp, sr * cp, cr * sp, -sr * sp]

    def _calibrate(self, gyro):
        if self.calibrated:
            return
        self.cal_window.append(tuple(gyro))
        if len(self.cal_window) < self.CAL_SAMPLES:
            return
        spread = max(max(row[i] for row in self.cal_window) -
                     min(row[i] for row in self.cal_window) for i in range(3))
        if spread < 3.0:
            self.bias = [sum(row[i] for row in self.cal_window) /
                         len(self.cal_window) for i in range(3)]
            self.cal_bias = list(self.bias)
            self.calibrated = True
        else:
            # Movement poisoned this window. Keep a short tail so calibration
            # restarts quickly as soon as the probe is actually held still.
            self.cal_window = self.cal_window[-5:]

    def update(self, accel, gyro, now=None):
        now = time.monotonic() if now is None else now
        ax, ay, az = (float(x) for x in accel)
        gx, gy, gz = (float(x) for x in gyro)

        if self.last_time is None:
            self._seed_from_gravity(ax, ay, az)
            self.last_time = now
            self._calibrate((gx, gy, gz))
            return tuple(self.q), self.still

        dt = now - self.last_time
        self.last_time = now
        if dt <= 0.0 or dt > 0.5:
            dt = 0.1
        self._calibrate((gx, gy, gz))

        rx, ry, rz = gx - self.bias[0], gy - self.bias[1], gz - self.bias[2]
        rate = math.sqrt(rx * rx + ry * ry + rz * rz)
        anorm = math.sqrt(ax * ax + ay * ay + az * az)
        quiet = (rate < self.STILL_GYRO_DPS and
                 abs(anorm - 1.0) < self.STILL_ACC_TOL)
        if not quiet:
            self.still_since = None
            self.still = False
        else:
            if self.still_since is None:
                self.still_since = now
            if now - self.still_since >= self.STILL_S:
                self.still = True
                if self.calibrated:
                    for i, raw in enumerate((gx, gy, gz)):
                        b = self.bias[i] + self.BIAS_LEARN * (raw - self.bias[i])
                        self.bias[i] = clamp(
                            b, self.cal_bias[i] - self.BIAS_CLAMP_DPS,
                            self.cal_bias[i] + self.BIAS_CLAMP_DPS)

        w, x, y, z = self.q
        wx, wy, wz = (math.radians(gx - self.bias[0]),
                      math.radians(gy - self.bias[1]),
                      math.radians(gz - self.bias[2]))
        if anorm > 0.5 and abs(anorm - 1.0) < 0.25:
            axn, ayn, azn = ax / anorm, ay / anorm, az / anorm
            vx = 2.0 * (x * z - w * y)
            vy = 2.0 * (w * x + y * z)
            vz = w * w - x * x - y * y + z * z
            ex = ayn * vz - azn * vy
            ey = azn * vx - axn * vz
            ez = axn * vy - ayn * vx
            self.integral[0] += self.KI * ex * dt
            self.integral[1] += self.KI * ey * dt
            self.integral[2] += self.KI * ez * dt
            wx += self.KP * ex + self.integral[0]
            wy += self.KP * ey + self.integral[1]
            wz += self.KP * ez + self.integral[2]

        dw = 0.5 * (-x * wx - y * wy - z * wz)
        dx = 0.5 * (w * wx + y * wz - z * wy)
        dy = 0.5 * (w * wy - x * wz + z * wx)
        dz = 0.5 * (w * wz + x * wy - y * wx)
        q = [w + dw * dt, x + dx * dt, y + dy * dt, z + dz * dt]
        n = math.sqrt(sum(v * v for v in q))
        if n > 1e-9:
            self.q = [v / n for v in q]
        return tuple(self.q), self.still


class WsJsonStream:
    """Small dependency-free WebSocket client for the official IMU endpoint."""

    def __init__(self, url, timeout=3.0):
        self.url = url
        self.timeout = timeout
        self.sock = None

    def connect(self):
        u = urlsplit(self.url)
        if u.scheme != "ws" or not u.hostname:
            raise ValueError("IMU URL must be ws://host/path")
        port = u.port or 80
        path = u.path or "/"
        if u.query:
            path += "?" + u.query
        sock = socket.create_connection((u.hostname, port), self.timeout)
        sock.settimeout(self.timeout)
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        req = ("GET {} HTTP/1.1\r\nHost: {}:{}\r\nUpgrade: websocket\r\n"
               "Connection: Upgrade\r\nSec-WebSocket-Key: {}\r\n"
               "Sec-WebSocket-Version: 13\r\n\r\n").format(
                   path, u.hostname, port, key)
        sock.sendall(req.encode("ascii"))
        header = bytearray()
        while b"\r\n\r\n" not in header:
            part = sock.recv(1)
            if not part or len(header) > 16384:
                raise ConnectionError("incomplete WebSocket handshake")
            header.extend(part)
        text_header = header.decode("iso-8859-1")
        if " 101 " not in text_header.split("\r\n", 1)[0]:
            raise ConnectionError("WebSocket upgrade rejected")
        expected = base64.b64encode(hashlib.sha1(
            (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11")
            .encode("ascii")).digest()).decode("ascii")
        if "sec-websocket-accept: " + expected.lower() not in text_header.lower():
            raise ConnectionError("invalid WebSocket accept key")
        self.sock = sock
        return self

    def _read_exact(self, n):
        out = bytearray()
        while len(out) < n:
            part = self.sock.recv(n - len(out))
            if not part:
                raise ConnectionError("WebSocket closed")
            out.extend(part)
        return bytes(out)

    def _send(self, opcode, payload=b""):
        payload = bytes(payload)
        first = 0x80 | opcode
        n = len(payload)
        if n < 126:
            head = bytes((first, 0x80 | n))
        elif n < 65536:
            head = bytes((first, 0x80 | 126)) + struct.pack("!H", n)
        else:
            head = bytes((first, 0x80 | 127)) + struct.pack("!Q", n)
        mask = os.urandom(4)
        masked = bytes(v ^ mask[i & 3] for i, v in enumerate(payload))
        self.sock.sendall(head + mask + masked)

    def recv_json(self):
        chunks = []
        opcode0 = None
        while True:
            b0, b1 = self._read_exact(2)
            final, opcode = bool(b0 & 0x80), b0 & 0x0f
            masked, n = bool(b1 & 0x80), b1 & 0x7f
            if n == 126:
                n = struct.unpack("!H", self._read_exact(2))[0]
            elif n == 127:
                n = struct.unpack("!Q", self._read_exact(8))[0]
            mask = self._read_exact(4) if masked else None
            payload = self._read_exact(n)
            if mask:
                payload = bytes(v ^ mask[i & 3] for i, v in enumerate(payload))
            if opcode == 0x8:
                raise ConnectionError("WebSocket close frame")
            if opcode == 0x9:
                self._send(0xA, payload)
                continue
            if opcode == 0xA:
                continue
            if opcode in (0x1, 0x2):
                opcode0 = opcode
                chunks = [payload]
            elif opcode == 0x0 and opcode0 is not None:
                chunks.append(payload)
            else:
                continue
            if final:
                if opcode0 != 0x1:
                    chunks, opcode0 = [], None
                    continue
                return json.loads(b"".join(chunks).decode("utf-8"))

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None


class OfficialProbeLink:
    """Stock M5Stack firmware: UVC video plus its IMU WebSocket."""

    is_uvc = True

    def __init__(self, video="auto", width=640, height=480,
                 imu_url="ws://192.168.4.1/api/v1/ws/imu_data"):
        self.camera = V4L2Source(video, width, height)
        self.imu_url = imu_url
        self.fusion = MahonyFusion()
        self.lock = threading.Lock()
        self.quat = (1.0, 0.0, 0.0, 0.0)
        self.still = False
        self.state = "searching"
        self.fw = "official_uvc"
        self.fw_ver = None
        self.calib_ok = None
        self.calibrating = True
        self.camera_failed = False
        self.generation = 0
        self.imu_time = 0.0
        self.bad_packets = 0
        self.running = False
        self.stream = None
        self.status = []
        self.raw_swap = False
        self.raw_stream = False
        self.test_pattern = False
        self.colour_mode = None
        self.sensor_preset = None
        self.sensor_regs = ""
        self.regs_seq = 0
        self.reg_ack = None

    def start(self):
        try:
            self.camera.start()
        except Exception as exc:
            self.camera_failed = True
            self.status.append({"status": "uvc_failed", "error": str(exc)})
        self.running = True
        threading.Thread(target=self._manager, daemon=True).start()
        return self

    def stop(self):
        self.running = False
        if self.stream:
            self.stream.close()
        self.camera.stop()

    def _manager(self):
        while self.running:
            with self.lock:
                self.state = "connecting"
            try:
                self.stream = WsJsonStream(self.imu_url).connect()
                self.fusion.reset()
                with self.lock:
                    self.state = "online"
                    self.generation += 1
                    self.calibrating = True
                    self.calib_ok = None
                while self.running:
                    data = self.stream.recv_json()
                    # Official SharedData::UpdateImuData publishes accel X/Y
                    # swapped but leaves gyro X/Y untouched. Undo only that
                    # published accel swap so both sensors share one body frame.
                    accel = (data["ay"], data["ax"], data["az"])
                    gyro = (data["gx"], data["gy"], data["gz"])
                    now = time.monotonic()
                    q, still = self.fusion.update(accel, gyro, now)
                    with self.lock:
                        self.quat = q
                        self.still = still
                        self.imu_time = now
                        self.calibrating = not self.fusion.calibrated
                        self.calib_ok = True if self.fusion.calibrated else None
            except (OSError, ValueError, KeyError, TypeError,
                    json.JSONDecodeError, ConnectionError):
                with self.lock:
                    self.state = "offline"
                self.bad_packets += 1
            finally:
                if self.stream:
                    self.stream.close()
                    self.stream = None
            if self.running:
                time.sleep(1.0)

    def snapshot(self):
        frame, seq, _ = self.camera.snapshot()
        with self.lock:
            return frame, seq, self.quat, self.still

    def health(self):
        now = time.monotonic()
        with self.lock:
            return {"state": self.state, "fw": self.fw,
                    "fw_ver": self.fw_ver, "colour_mode": None,
                    "sensor_preset": None, "sensor_regs": "",
                    "regs_seq": 0, "reg_ack": None, "raw_stream": False,
                    "test_pattern": False, "calib_ok": self.calib_ok,
                    "raw_seq": 0, "calibrating": self.calibrating,
                    "camera_failed": self.camera_failed,
                    "generation": self.generation,
                    "imu_age": now - self.imu_time if self.imu_time else 1e9,
                    "frame_age": (now - self.camera.frame_time
                                  if self.camera.frame_time else 1e9),
                    "fps": self.camera.fps(), "bad": self.bad_packets,
                    # The stock Wi-Fi route publishes no device clock, so gap
                    # accounting is unavailable -- reported as -1 rather than
                    # 0, which would falsely read as "no losses".
                    "imu_gaps": -1, "imu_gap_ms": -1.0, "worst_gap_ms": -1.0,
                    "port": self.imu_url}

    def send_bytes(self, _):
        return False

    def send_byte(self, _):
        return False

    def get_regs(self):
        return {}, None

    def take_raw(self):
        return None, 0


class SimLink:
    """
    Synthetic probe with the same interface as ProbeLink: a scripted attitude
    (slow heading sweep, occasional tilt, pauses) and a generated test-pattern
    video. Lets the whole app -- stages, indicator, HUD, drift readout -- run
    and be exercised on any machine with no hardware attached.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.camera = None
        self.frame = None
        self.frame_seq = 0
        self.quat = (1.0, 0.0, 0.0, 0.0)
        self.still = False
        self.state = "online"
        self.fw = "ready"
        self.fw_ver = (3, 8)
        self.colour_mode = 1
        self.sensor_preset = (0, "rgb565 driver default")
        self.sensor_regs = "14=10 22=57 24=a6 54=22 (sim)"
        self._regs = {0x14: 0x10, 0x22: 0x57, 0x24: 0xa6, 0x03: 0x01, 0x04: 0xe8,
                      0x50: 0x14, 0x5a: 0x56, 0x5b: 0x40, 0x5c: 0x4a, 0xb1: 0x40,
                      0xb2: 0x40, 0xb3: 0x40, 0xb4: 0x80, 0xb5: 0x00}
        self.regs_dict = dict(self._regs)
        self.regs_seq = 1
        self.defaults = dict(self._regs)
        self.reg_ack = None
        self._ack_n = 0
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

    def send_bytes(self, bs):
        bs = bytes(bs)
        if len(bs) == 3 and bs[0] == ord("W"):
            with self.lock:
                self._regs[bs[1]] = bs[2]
                self._ack_n += 1
                self.reg_ack = (self._ack_n, bs[1], bs[2], bs[2], True)
            return True
        if bs[:1] in (b"g", b"G"):
            with self.lock:
                self.regs_dict = dict(self._regs)
                self.regs_seq += 1
            return True
        return self.send_byte(bs[0]) if bs else False

    def get_regs(self):
        with self.lock:
            return dict(self.regs_dict), dict(self.defaults)

    def send_byte(self, b):
        if b in (ord("c"), ord("C")):
            with self.lock:
                self.colour_mode = (self.colour_mode + 1) % 3
        elif b in (ord("0"), ord("1"), ord("2")):
            with self.lock:
                self.colour_mode = b - ord("0")
        elif b in (ord("n"), ord("N"), ord("p"), ord("P")):
            with self.lock:
                n = (self.sensor_preset[0] + (1 if b in (ord("n"), ord("N"))
                                              else 8)) % 9
                self.sensor_preset = (n, "sim preset %d" % n)
                self.colour_mode = 2 if n in (1, 2) else 0
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
                    "sensor_preset": self.sensor_preset,
                    "sensor_regs": self.sensor_regs,
                    "regs_seq": self.regs_seq, "reg_ack": self.reg_ack,
                    "calib_ok": self.calib_ok, "raw_seq": self.raw_seq,
                    "camera_failed": False, "generation": self.generation,
                    "imu_age": now - self.imu_time if self.imu_time else 1e9,
                    "frame_age": ((now - self.camera.frame_time)
                              if self.camera is not None and self.camera.frame_time
                              else (now - self.frame_time if self.frame_time else 1e9)),
                    "fps": 12.5, "bad": 0, "port": "sim",
                    "imu_gaps": 0, "imu_gap_ms": 0.0, "worst_gap_ms": 0.0}


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
    STAGE_TUNE = 3       # manual sensor tuning: rows left, live picture right
    STAGE_CAL = 4        # guided calibration: the screen is the colour target

    def __init__(self, link, args):
        self.link = link
        self.args = args
        self.cfg = self.load_cfg()
        self.clean_video = bool(getattr(link, "is_uvc", False) or
                                getattr(link, "camera", None) is not None)
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
        self.rot180 = bool(self.cfg.get("rot180", False))
        # ---- operator display flips (v6) -------------------------------
        # These mirror the PICTURE ONLY. They are applied at the very end of
        # the render path, after every colour step and after rot180, and they
        # are never fed back into aim_angles() or the indicator. The probe's
        # body axes are a physical fact; letting a display button rotate them
        # is exactly how v3.1 ended up saving a wrong axis. See HANDOFF S4.
        self.flip_btn = {}          # {"flip_h": (rect_id, text_id), ...}
        self.mirror_flag = None     # persistent "MIRRORED" HUD warning
        self.flip_h = bool(self.cfg.get("flip_h", False))   # mirror left/right
        self.flip_v = bool(self.cfg.get("flip_v", False))   # mirror up/down
        # Up/down polarity, set once with FLIP U/D and never touched
        # by zeroing or by the axis detector.
        # Default is INVERTED: on this build of the probe the lens axis the
        # detector locks is the antipode of the true one, so a lift read as
        # a drop until FLIP U/D was pressed every session. Baked in; the
        # button stays so a differently-mounted probe can undo it.
        self.el_sign = 1.0 if self.cfg.get("el_sign", -1) > 0 else -1.0
        self.awb = bool(self.cfg.get("awb", False))
        self.tune_saved = (self.cfg.get("tune")
                           if args.legacy_colour_tools else None)
        cal = self.cfg.get("cal") or {}
        self.pi_gains = [float(x) for x in cal.get("gains", [1.0, 1.0, 1.0])]  # B,G,R
        self.pi_chan = [int(x) for x in cal.get("chan", [0, 1, 2])]            # BGR index map
        # Black point, white point and gamma, per channel. Gains alone cannot
        # fix this sensor: measured on a real frame, blue sits only 22 below
        # the others at the 99th percentile but 100 below at the median. A
        # gain is a straight multiply, so lifting blue's midtones by the 3x
        # they need drives its highlights past 255 and the whites burn out --
        # which is exactly what "no black and no white" looks like. Anchoring
        # black, white and mid separately is the only thing that fixes a
        # transfer curve rather than a level.
        self.pi_black = [float(x) for x in cal.get("black", [0.0, 0.0, 0.0])]
        self.pi_white = [float(x) for x in cal.get("white", [255.0, 255.0, 255.0])]
        self.pi_gamma = [float(x) for x in cal.get("gamma", [1.0, 1.0, 1.0])]
        self._lut = None
        self.cal_group = 0
        self.cal_prim = {}      # measured R,G,B primaries for the matrix
        self.pi_bright = float(cal.get("bright", 0.0))    # -80 .. +80, added
        self.pi_sat = float(cal.get("sat", 1.0))          # 0 .. 2, multiplies
        self.pi_matrix = cal.get("matrix")           # 4x3 colour correction, or None
        self._bars_pending = False
        self.cal_meas = None
        self.cal_sign = 1
        self.cal_perm = {}
        self.cal_thumb = None
        self.cal_thumb_photo = None
        self.cal_thumb_dims = (160, 120)
        self.cal_text = None
        self.tune = {"regs": {}, "cm": "0", "poke_reg": 0x93, "poke_val": 0x40}
        self.tune_page = 0
        self.tune_vals = {}
        self.tune_video = None
        self.tune_photo = None
        self.tune_line = None
        self.tune_box = (0, 0, 1, 1)
        self._tune_pop_seq = -1
        self._tune_ack_seq = 0
        self._tune_gen_applied = -1
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
                cfg = json.load(f)
            if int(cfg.get("config_rev", 0)) == CONFIG_REV:
                return cfg
            # v4.x persisted experimental sensor registers, colour matrices,
            # grey-world AWB and a 180-degree compensation. Replaying those
            # settings is enough to corrupt a now-correct official frame, so
            # migrate only the proven orientation choices.
            # v5 and earlier had no independent flips. Orientation choices
            # (which body axis the lens looks along, and the up/down polarity)
            # are hard-won and must survive; everything colour-related is
            # deliberately dropped, and the flips start off.
            return {"config_rev": CONFIG_REV,
                    "axis": int(cfg.get("axis", 0)),
                    "el_sign": float(cfg.get("el_sign", -1)),
                    "rot180": False, "awb": False, "swap_rb": False,
                    "flip_h": False, "flip_v": False}
        except Exception:
            return {"config_rev": CONFIG_REV}

    def save_cfg(self):
        try:
            os.makedirs(os.path.dirname(CONFIG), exist_ok=True)
            with open(CONFIG, "w", encoding="utf-8") as f:
                json.dump({"config_rev": CONFIG_REV,
                           "axis": self.axis_idx,
                           "rot180": self.rot180,
                           "flip_h": self.flip_h,
                           "flip_v": self.flip_v,
                           "el_sign": self.el_sign,
                           "awb": self.awb,
                           "swap_rb": self.swap_rb,
                           "tune": self.tune_saved,
                           "cal": {"gains": self.pi_gains, "chan": self.pi_chan,
                                   "bright": self.pi_bright, "sat": self.pi_sat,
                                   "black": self.pi_black,
                                   "white": self.pi_white,
                                   "gamma": self.pi_gamma,
                                   "matrix": self.pi_matrix}}, f)
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
        elif stage == self.STAGE_TUNE:
            self.enter_tune()
        elif stage == self.STAGE_CAL:
            self.enter_cal()
        else:
            self.enter_run()

    # ---- zero capture, shared by every stage

    def link_ready(self):
        h = self.link.health()
        return (h["state"] == "online" and h["imu_age"] < 1.0
                and not h.get("calibrating", False)
                and h.get("calib_ok") is not False)

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
        if self.args.legacy_colour_tools and not self.clean_video:
            self.button(pad, H - pad - int(H * 0.055), "COLOR", "#455a64",
                        self.cycle_colour, max(10, int(H / 38)),
                        store=self.hud, anchor="sw")
            self.button(pad, H - pad - int(H * 0.055) - int(H * 0.085), "DIAG",
                        "#4e5d3a", self.toggle_diag, max(10, int(H / 38)),
                        store=self.hud, anchor="sw")
            self.button(pad, H - pad - int(H * 0.055) - int(H * 0.170), "TUNE",
                        "#5d4037", lambda: self.set_stage(self.STAGE_TUNE),
                        max(10, int(H / 38)), store=self.hud, anchor="sw")
        # Sensor register dump from the last SENSOR press: lets the chip's
        # real state be read off a photograph of the screen.
        self.regs_item = self.canvas.create_text(
            W * 0.21, H - pad, anchor="sw", text="", fill="#9ccc65",
            font=self.f(max(9, int(H / 54))))
        self.hud.append(self.regs_item)

        # ---- display flip buttons (v6) ---------------------------------
        # Bottom-right, stacked above ZERO so the thumb reaches all three
        # without crossing the picture. They only ever touch the displayed
        # frame; the indicator keeps reading the probe's real attitude.
        fbs = max(11, int(H / 36))
        zero_h = fbs * 2 + 30
        fy = H - pad - zero_h - int(H * 0.020)
        r, t, fw_btn, fh_btn = self.button(
            W - pad, fy, "FLIP V", FLIP_ON if self.flip_v else FLIP_OFF,
            self.toggle_flip_v, fbs, store=self.hud, anchor="se")
        self.flip_btn["flip_v"] = (r, t)
        r, t, _, _ = self.button(
            W - pad, fy - fh_btn - int(H * 0.018), "FLIP H",
            FLIP_ON if self.flip_h else FLIP_OFF,
            self.toggle_flip_h, fbs, store=self.hud, anchor="se")
        self.flip_btn["flip_h"] = (r, t)

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
        # A mirrored picture moves opposite to the arrow. That is dangerous
        # while steering, so the warning sits ON the indicator panel, where
        # the operator is already looking, not in the status line.
        self.mirror_flag = c.create_text(
            px + panel / 2, py + panel * 0.99, text="", fill=WARN,
            font=self.f(max(8, panel // 19), True), state="hidden")
        self.hud.append(self.mirror_flag)
        self.statusbar = c.create_text(pad, H - pad, anchor="sw", text="",
                                       fill=MUTED,
                                       font=self.f(max(8, H // 62)))
        self.hud.append(self.statusbar)
        self.nosignal = c.create_text(W / 2, H / 2, text="NO VIDEO SIGNAL",
                                      fill=MUTED,
                                      font=self.f(max(15, int(H / 20)), True),
                                      state="hidden")
        self.hud.append(self.nosignal)
        # Flips persist across restarts, so paint the buttons and the mirror
        # warning to match what was restored from the config file.
        self._refresh_flip_labels()

    def _apply_flips(self, img):
        """
        Mirror the displayed frame. Picture only -- never the attitude.

        cv2.flip codes: 1 = about the vertical axis (left/right mirror),
        0 = about the horizontal axis (up/down), -1 = both, which is the same
        result as a 180-degree rotation. Doing both in one call is cheaper
        than two passes over the buffer, which matters on a Pi pushing frames
        at video rate.
        """
        if self.flip_h and self.flip_v:
            return cv2.flip(img, -1)
        if self.flip_h:
            return cv2.flip(img, 1)
        if self.flip_v:
            return cv2.flip(img, 0)
        return img

    def toggle_flip_h(self):
        """
        Mirror left/right.

        WARNING SHOWN TO THE OPERATOR: a horizontal mirror decouples the
        picture from the direction indicator. The probe still physically
        turns right when the arrow says right, but a mirrored image will
        appear to move LEFT. That is a real hazard when steering by video, so
        the HUD carries a persistent marker while this is on. A vertical flip
        does not have this problem for left/right, so it gets no marker.
        """
        self.flip_h = not self.flip_h
        self.save_cfg()
        self._refresh_flip_labels()
        if self.flip_h:
            self.toast("MIRRORED \u2014 image left/right is reversed "
                       "vs the arrow", WARN, ms=2600)
        else:
            self.toast("Mirror off", OK)

    def toggle_flip_v(self):
        """Flip the picture up/down. Does not affect the indicator."""
        self.flip_v = not self.flip_v
        self.save_cfg()
        self._refresh_flip_labels()
        self.toast("Flip V on" if self.flip_v else "Flip V off", OK)

    def _refresh_flip_labels(self):
        """Repaint the two flip buttons so their state is visible at a glance."""
        for key, on in (("flip_h", self.flip_h), ("flip_v", self.flip_v)):
            item = self.flip_btn.get(key)
            if not item:
                continue
            rect, label = item
            try:
                self.canvas.itemconfigure(rect, fill=(FLIP_ON if on
                                                      else FLIP_OFF))
                self.canvas.itemconfigure(label, fill=("#0d1117" if on
                                                       else "#ffffff"))
            except Exception:
                pass
        # Persistent mirror warning next to the indicator.
        if getattr(self, "mirror_flag", None):
            self.canvas.itemconfigure(
                self.mirror_flag,
                text="MIRRORED" if self.flip_h else "",
                state="normal" if self.flip_h else "hidden")

    def toggle_test_pattern(self):
        """
        Put a known image into the pipeline instead of the sensor's.

        Colour bars and a grey ramp are generated in the probe and travel the
        identical path a camera frame takes. Because the input is known, the
        output is diagnostic on its own: clean bars mean the pipeline is
        sound and the sensor's pixels are the problem; wrong bars name the
        fault by the way they are wrong.
        """
        if not self.link.send_byte(ord("t")):
            self.toast("PROBE NOT CONNECTED", WARN)
            return
        self.toast("TEST PATTERN toggled \u2014 expect 8 colour bars "
                   "over a grey ramp", OK, ms=3200)

    def toggle_raw_stream(self):
        """
        Ask the probe to stop encoding and just send its bytes.

        This is the deciding test for the colour argument. In this mode no
        firmware code touches a pixel: the sensor's RGB565 travels untouched
        and OpenCV's own cvtColor turns it into a picture here. If it looks
        right, the fault was always in the firmware's conversion. If it still
        looks wrong, pixel interpretation was never the problem and the search
        moves to the parallel bus.

        Costs resolution -- 160x120, since raw is four times the size of the
        JPEG and the IMU shares the link -- which is a fair price for an
        answer.
        """
        if not self.link.send_byte(ord("s")):
            self.toast("PROBE NOT CONNECTED", WARN)
            return
        self.toast("RAW STREAM toggled \u2014 no firmware colour conversion",
                   OK, ms=2600)

    def cycle_colour(self):
        """
        Ask the probe for its next colour mode (0 raw / 1 byte-swap / 2 YUV).
        A scrambled picture is fixed by pressing this until it looks natural:
        the failure is in how the sensor's bytes are interpreted before JPEG,
        so it can only be fixed probe-side, and cycling live beats reflashing
        once per guess. The probe acknowledges with a status packet, which
        pops the COLOUR MODE toast below.
        """
        # No version gate. fw_ver is only learned from the boot status line,
        # so a Pi that attached after the probe had already booted never sees
        # it and would refuse a perfectly capable probe -- which is exactly
        # what happened in the field. Send the byte; an old probe simply
        # ignores it.
        if not self.link.send_byte(ord("c")):
            self.toast("PROBE NOT CONNECTED", WARN)

    def next_preset(self, back=False):
        """
        Step the probe's sensor register preset (see SENSOR PRESETS in the
        firmware). The Pi-side white balance is switched off first: it would
        mask exactly the differences we are trying to see. Press until the
        picture looks natural, then report the preset number.
        """
        h = self.link.health()
        fv = h.get("fw_ver")
        if fv is None or fv < (3, 7):
            self.toast("PROBE FIRMWARE HAS NO SENSOR PRESETS \u2014 FLASH v3.9",
                       WARN, ms=2600)
            return
        if getattr(self, "awb", False):
            self.awb = False
            self.save_cfg()
        if not self.link.send_byte(ord("p" if back else "n")):
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
        if (k in ("c", "d", "n", "p", "g", "r", "x", "b", "w", "t")
                and (not self.args.legacy_colour_tools or self.clean_video)):
            if self.stage == self.STAGE_RUN:
                self.toast("OFFICIAL VIDEO PATH — colour overrides disabled",
                           OK, ms=1400)
            return
        # v6: display flips. Deliberately NOT in the colour-override
        # block above -- flipping is a legitimate operator control on
        # the official video path, not a colour experiment.
        if k == "h" and self.stage == self.STAGE_RUN:
            self.toggle_flip_h()
            return
        if k == "j" and self.stage == self.STAGE_RUN:
            self.toggle_flip_v()
            return
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
        elif k == "n" and self.stage == self.STAGE_RUN:
            self.next_preset()
        elif k == "t" and self.stage == self.STAGE_RUN:
            self.set_stage(self.STAGE_TUNE)
        elif k in ("return", "kp_enter") and self.stage == self.STAGE_TUNE:
            self.tune_done()
        elif k == "p" and self.stage == self.STAGE_RUN:
            self.next_preset(back=True)
        elif k == "g" and self.stage == self.STAGE_RUN:
            self.toggle_test_pattern()
        elif k == "r" and self.stage == self.STAGE_RUN:
            self.toggle_raw_stream()
        elif k == "x" and self.stage == self.STAGE_RUN:
            self.link.raw_swap = not self.link.raw_swap
            self.toast("RAW BYTE ORDER: "
                       + ("SWAPPED" if self.link.raw_swap else "NORMAL"),
                       OK, ms=1400)
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
        if getattr(self.link, "is_uvc", False):
            if h["state"] in ("searching", "connecting"):
                return ("PROBE: connect Pi Wi-Fi to AtomS3R-CAM-WiFi; "
                        "waiting for IMU...", DIM)
            if h["state"] == "offline":
                return ("PROBE: IMU link offline — check AtomS3R-CAM-WiFi "
                        "and 192.168.4.1", ALERT)
            if h["camera_failed"]:
                return ("PROBE: IMU online, UVC missing — check /dev/video*",
                        WARN)
            if h["calibrating"]:
                return ("PROBE: official UVC online; calibrating gyro — "
                        "hold it still...", WARN)
            return "PROBE: official UVC + IMU online", OK
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

        # The sensor forgets everything on power-up. Whatever the operator
        # DONE'd in TUNE is replayed on every new probe boot, once its own
        # defaults dump has landed.
        if (self.args.legacy_colour_tools and not self.clean_video
                and self.tune_saved and h.get("fw_ver") is not None
                and h.get("fw_ver", (0, 0)) >= (3, 10)
                and h["generation"] != self._tune_gen_applied):
            self._tune_gen_applied = h["generation"]
            self.root.after(1500, self._tune_apply_saved)

        if self.stage == self.STAGE_TUNE:
            self._tune_update(h, frame, seq)
            self.root.after(40, self.update)
            return
        if self.stage == self.STAGE_CAL:
            self._cal_update(h, frame, seq)
            self.root.after(40, self.update)
            return

        if self.stage == self.STAGE_SETUP:
            if self.setup_status:
                text, col = self.probe_status_text(h)
                self.canvas.itemconfigure(self.setup_status,
                                          text=text, fill=col)
            if self.zero_btn:
                ready = (h["state"] == "online" and h["imu_age"] < 1.0
                         and not h.get("calibrating", False)
                         and h.get("calib_ok") is not False)
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
            uvc = self.clean_video
            cm = h.get("colour_mode")
            if cm is not None and cm != self._last_cm:
                prev, self._last_cm = self._last_cm, cm
                if prev is not None:      # not the first report after boot
                    names = {0: "OFFICIAL", 1: "LEGACY RGB565", 2: "YUV"}
                    self.toast("COLOUR MODE {} \u2014 {}".format(
                        cm, names.get(cm, "?")), OK, ms=1600)
            regs = h.get("sensor_regs") or ""
            if regs and regs != getattr(self, "_last_regs", None) and \
                    getattr(self, "regs_item", None):
                self._last_regs = regs
                self.canvas.itemconfigure(self.regs_item, text="SENSOR REGS  " + regs)
            sp = h.get("sensor_preset")
            if sp is not None and sp != getattr(self, "_last_sp", None):
                prev_sp = getattr(self, "_last_sp", None)
                self._last_sp = sp
                if prev_sp is not None and sp[1]:
                    # Wins over the colour toast fired just above: the preset
                    # is the thing the operator pressed for.
                    self.toast("SENSOR PRESET {} \u2014 {}".format(sp[0], sp[1]),
                               OK, ms=2600)
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
                        # The raw stream needs the same answer: mode 0 is
                        # high-byte-first, which OpenCV needs swapped.
                        if best in (0, 1):
                            self.link.raw_swap = (best == 0)
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
                if self.swap_rb and self.diag_photo is None and not uvc:
                    # A red/blue transposition survives JPEG intact, so unlike
                    # a bit-order fault it can still be undone here.
                    small = small[:, :, ::-1]
                if self.diag_photo is None and not uvc:
                    small = self._pi_colour(small)
                if self.awb and self.diag_photo is None and not uvc:
                    small, self._awb_gains = grey_world(small, self._awb_gains)
                # Operator flips are the LAST thing done to the picture, and
                # only to the picture. DIAG output is excluded on purpose: it
                # exists to show the sensor's unaltered truth, and a mirrored
                # diagnostic would be worse than none.
                if self.diag_photo is None:
                    small = self._apply_flips(small)
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
            if uvc:
                bits.append("OFFICIAL UVC")
            elif self.awb:
                bits.append("AWB")
            if h.get("raw_stream"):
                bits.append("RAW" + ("~" if self.link.raw_swap else ""))
            if h.get("test_pattern"):
                bits.append("TESTPAT")
            if self.diag_photo is not None:
                bits.append("DIAG")
            sp = h.get("sensor_preset")
            if sp is not None:
                bits.append("SP{}".format(sp[0]))
            if h["state"] != "online":
                bits.append(h["state"].upper())
            if still:
                bits.append("bias trim")
            if h["bad"]:
                bits.append("{} bad pkts".format(h["bad"]))
            self.canvas.itemconfigure(self.statusbar, text="    ".join(bits))

        self.root.after(40, self.update)


    # ---- TUNE stage: every image-affecting knob, live, saved on DONE -------

    TUNE_FORMATS = [("RGB565", 0xa6, "0"), ("YUV Y-Cb a2", 0xa2, "2"),
                    ("YUV Y-Cr a3", 0xa3, "2"), ("YUV Cb-Y a0", 0xa0, "3"),
                    ("YUV Cr-Y a1", 0xa1, "3")]
    TUNE_FLIPS = [("none", 0x10), ("mirror", 0x11), ("flip", 0x12), ("both", 0x13)]
    TUNE_REGS = [0x24, 0x22, 0x14, 0x03, 0x04, 0x50, 0x5a, 0x5b, 0x5c,
                 0xb1, 0xb2, 0xb3, 0xb5]
    # (key, label, kind, spec)
    TUNE_PAGES = [
        [("fmt",  "OUTPUT FORMAT",   "fmt",   None),
         ("aec",  "AUTO EXPOSURE",   "bit",   0x01),
         ("awb",  "AUTO WHITE BAL",  "bit",   0x02),
         ("agc",  "AUTO GAIN",       "bit",   0x04),
         ("r",    "WB GAIN  R",      "reg",   (0x5a, 4, 16, 0x02)),
         ("g",    "WB GAIN  G",      "reg",   (0x5b, 4, 16, 0x02)),
         ("b",    "WB GAIN  B",      "reg",   (0x5c, 4, 16, 0x02)),
         ("satb", "SATURATION Cb",   "reg",   (0xb1, 4, 16, None)),
         ("satr", "SATURATION Cr",   "reg",   (0xb2, 4, 16, None)),
         ("con",  "CONTRAST",        "reg",   (0xb3, 4, 16, None)),
         ("bri",  "BRIGHTNESS",      "sreg",  (0xb5, 4, 16, None))],
        [("exp",  "EXPOSURE 03/04",  "reg16", (0x03, 0x04, 32, 256, 0x01)),
         ("gain", "GLOBAL GAIN 50",  "reg",   (0x50, 4, 16, 0x04)),
         ("flip", "FLIP / MIRROR",   "flip",  None),
         ("rot",  "PI: ROTATE 180",  "pi",    "rot180"),
         ("pawb", "PI: SOFT AWB",    "pi",    "awb"),
         ("prb",  "PI: SWAP R/B",    "pi",    "swap_rb"),
         ("preg", "POKE  REGISTER",  "poke_reg", None),
         ("pval", "POKE  VALUE",     "poke_val", None)],
    ]

    def enter_tune(self):
        c, W, H, z = self.canvas, self.W, self.H, self.stage_items
        pw = int(W * 0.50)
        pad = max(10, H // 46)
        small = max(10, int(H / 48))
        z.append(c.create_rectangle(0, 0, pw, H, fill="#0e161b", outline=""))
        z.append(c.create_text(12, 10, anchor="nw", fill="white",
                               text="TUNE   page {}/{}".format(
                                   self.tune_page + 1, len(self.TUNE_PAGES)),
                               font=self.f(small + 3, True)))
        z.append(c.create_text(W - 8, 14, anchor="ne", fill=MUTED,
                               text="adjust left \u00b7 watch right \u00b7 DONE saves \u00b7 screenshot the green line",
                               font=self.f(small - 1)))

        rows = self.TUNE_PAGES[self.tune_page]
        y0, y1 = int(H * 0.08), H - int(H * 0.18)
        rh = (y1 - y0) // max(1, len(rows))
        # Everything is measured from the actual font so the columns fit any
        # panel size (a 7-inch 800x480 is narrower than it looks). Shrink the
        # font until the row fits.
        samples = ["0x00  (000)", "YUV Cb-Y a0", "-128", "reg 0x00", "4095"]
        while True:
            lab_w = max(self.text_w(r[1], small) for r in rows) + 10
            bw = self.text_w("<<", small, True) + 14
            val_w = max(self.text_w(s, small, True) for s in samples) + 12
            need = 12 + lab_w + 4 * (bw + 4) + val_w + 6
            if need <= pw or small <= 8:
                break
            small -= 1
        bh = min(rh - 6, int(H * 0.07))
        self.tune_vals = {}
        for i, (key, label, kind, spec) in enumerate(rows):
            y = y0 + i * rh
            yb = y + (rh - bh) // 2
            z.append(c.create_text(12, y + rh / 2, anchor="w", text=label,
                                   fill="#cfd8dc", font=self.f(small)))
            x = 12 + lab_w
            for lab, d in (("<<", -2), ("<", -1)):
                self._tune_btn(x, yb, bw, bh, lab,
                               lambda k=key, d=d: self._tune_step(k, d))
                x += bw + 4
            self.tune_vals[key] = c.create_text(x + val_w / 2, y + rh / 2,
                                                text="?", fill="white",
                                                font=self.f(small, True))
            z.append(self.tune_vals[key])
            x += val_w
            for lab, d in ((">", 1), (">>", 2)):
                self._tune_btn(x, yb, bw, bh, lab,
                               lambda k=key, d=d: self._tune_step(k, d))
                x += bw + 4

        # live picture, right side
        bx0, by0 = pw + 8, int(H * 0.08)
        bx1, by1 = W - 8, H - int(H * 0.18)
        self.tune_box = (bx0, by0, bx1 - bx0, by1 - by0)
        z.append(c.create_rectangle(bx0, by0, bx1, by1, outline=EDGE, width=1))
        self.tune_video = c.create_image((bx0 + bx1) // 2, (by0 + by1) // 2)
        z.append(self.tune_video)
        self.tune_photo = None
        self.last_seq = -1

        # values line: the thing to screenshot
        self.tune_line = c.create_text(12, H - int(H * 0.14), anchor="w", text="",
                                       fill="#9ccc65", font=self.f(max(10, int(H / 50)), True))
        z.append(self.tune_line)

        bs = max(12, int(H / 34))
        self.button(12, H - pad, "RESET", "#455a64", self.tune_reset, bs,
                    store=z, anchor="sw")
        self.button(12 + int(W * 0.13), H - pad, "PAGE", "#37474f",
                    self.tune_next_page, bs, store=z, anchor="sw")
        self.button(12 + int(W * 0.26), H - pad, "CAL WIZARD", "#00695c",
                    lambda: self.set_stage(self.STAGE_CAL), bs, store=z, anchor="sw")
        self.button(12 + int(W * 0.44), H - pad, "BARS TEST", "#4e342e",
                    self.tune_bars, bs, store=z, anchor="sw")
        self.button(W - pad, H - pad, "DONE  \u2713", "#2e7d32", self.tune_done,
                    bs, store=z, anchor="se")

        self._tune_pop_seq = -1              # take the next dump
        self.link.send_bytes(b"g")
        self._tune_refresh()

    def _tune_btn(self, x, y, w, h, label, cb):
        c = self.canvas
        r = c.create_rectangle(x, y, x + w, y + h, fill="#37474f", outline="")
        t = c.create_text(x + w / 2, y + h / 2, text=label, fill="white",
                          font=self.f(max(10, int(self.H / 48)), True))
        for item in (r, t):
            c.tag_bind(item, "<Button-1>", lambda e: cb())
        self.stage_items.extend((r, t))

    def tune_bars(self):
        """Sensor colour bars on -> one raw frame -> automatic verdict."""
        h = self.link.health()
        fv = h.get("fw_ver")
        if fv is None or fv < (3, 11):
            self.toast("FLASH FIRMWARE v3.11 FOR THE BARS TEST", WARN, ms=2400)
            return
        self._bars_pending = True
        self.link.send_bytes(b"b")
        self.root.after(900, lambda: self.link.send_bytes(b"r"))
        self.toast("SENSOR BARS: capturing \u2026", DIM, ms=1500)

    def tune_next_page(self):
        self.tune_page = (self.tune_page + 1) % len(self.TUNE_PAGES)
        self.set_stage(self.STAGE_TUNE)

    # ---- values

    def _reg(self, reg, default=0):
        return self.tune["regs"].get(reg, default)

    def _tune_write(self, reg, val):
        val = max(0, min(255, int(val)))
        self.tune["regs"][reg] = val
        if not self.link.send_bytes(bytes([ord("W"), reg, val])):
            self.toast("PROBE NOT CONNECTED", WARN)
        self._tune_refresh()

    def _tune_send_cm(self):
        self.link.send_bytes(self.tune["cm"].encode())

    def _tune_clear_auto(self, mask, what):
        v = self._reg(0x22, 0x57)
        if v & mask:
            self._tune_write(0x22, v & ~mask)
            self.toast("AUTO {} switched OFF for manual control".format(what),
                       DIM, ms=1400)

    def _tune_step(self, key, d):
        rows = {k: (kind, spec) for page in self.TUNE_PAGES
                for (k, _, kind, spec) in page}
        kind, spec = rows[key]
        sign = 1 if d > 0 else -1
        big = abs(d) == 2
        if kind == "fmt":
            codes = [f[1] for f in self.TUNE_FORMATS]
            cur = self._reg(0x24, 0xa6)
            i = codes.index(cur) if cur in codes else 0
            name, code, cm = self.TUNE_FORMATS[(i + sign) % len(codes)]
            self.tune["cm"] = cm
            self._tune_write(0x24, code)
            self._tune_send_cm()
        elif kind == "bit":
            self._tune_write(0x22, self._reg(0x22, 0x57) ^ spec)
        elif kind in ("reg", "sreg"):
            reg, s, b, auto = spec
            step = (b if big else s) * sign
            cur = self._reg(reg, 0x40)
            if kind == "sreg":
                cur = cur - 256 if cur > 127 else cur
                cur = max(-128, min(127, cur + step)) & 0xff
            else:
                cur = max(0, min(255, cur + step))
            if auto:
                self._tune_clear_auto(auto, {0x02: "WHITE BAL", 0x04: "GAIN",
                                             0x01: "EXPOSURE"}[auto])
            self._tune_write(reg, cur)
        elif kind == "reg16":
            hi, lo, s, b, auto = spec
            cur = (self._reg(hi, 1) << 8) | self._reg(lo, 0xe8)
            cur = max(1, min(4095, cur + (b if big else s) * sign))
            if auto:
                self._tune_clear_auto(auto, "EXPOSURE")
            self.tune["regs"][hi] = cur >> 8
            self._tune_write(hi, cur >> 8)
            self._tune_write(lo, cur & 0xff)
        elif kind == "flip":
            codes = [f[1] for f in self.TUNE_FLIPS]
            cur = self._reg(0x14, 0x10)
            i = codes.index(cur) if cur in codes else 0
            self._tune_write(0x14, codes[(i + sign) % len(codes)])
        elif kind == "pi":
            setattr(self, spec, not getattr(self, spec))
            self.save_cfg()
            self.last_seq = -1
            self._tune_refresh()
        elif kind == "poke_reg":
            self.tune["poke_reg"] = (self.tune["poke_reg"]
                                     + (16 if big else 1) * sign) & 0xff
            self._tune_refresh()
        elif kind == "poke_val":
            self.tune["poke_val"] = (self.tune["poke_val"]
                                     + (16 if big else 1) * sign) & 0xff
            self._tune_write(self.tune["poke_reg"], self.tune["poke_val"])

    def _tune_value_text(self, key, kind, spec):
        if kind == "fmt":
            cur = self._reg(0x24, None)
            for name, code, _ in self.TUNE_FORMATS:
                if code == cur:
                    return name
            return "0x%02X ?" % cur if cur is not None else "?"
        if kind == "bit":
            v = self._reg(0x22, None)
            return "?" if v is None else ("ON" if v & spec else "OFF")
        if kind == "reg":
            v = self._reg(spec[0], None)
            return "?" if v is None else "0x%02X  (%d)" % (v, v)
        if kind == "sreg":
            v = self._reg(spec[0], None)
            if v is None:
                return "?"
            return "%+d" % (v - 256 if v > 127 else v)
        if kind == "reg16":
            hi, lo = self._reg(spec[0], None), self._reg(spec[1], None)
            return "?" if hi is None or lo is None else str((hi << 8) | lo)
        if kind == "flip":
            cur = self._reg(0x14, None)
            for name, code in self.TUNE_FLIPS:
                if code == cur:
                    return name
            return "?"
        if kind == "pi":
            return "ON" if getattr(self, spec) else "OFF"
        if kind == "poke_reg":
            return "reg 0x%02X" % self.tune["poke_reg"]
        if kind == "poke_val":
            return "0x%02X" % self.tune["poke_val"]
        return "?"

    def _tune_values_line(self):
        regs = " ".join("%02x=%02x" % (r, self._reg(r, 0))
                        for r in self.TUNE_REGS if r in self.tune["regs"])
        return "TUNE  {}  cm={}  |  pi rot={:d} awb={:d} rb={:d} gains B{:.2f} R{:.2f}".format(
            regs or "(waiting for probe)", self.tune["cm"],
            self.rot180, self.awb, self.swap_rb, self.pi_gains[0], self.pi_gains[2])

    def _tune_refresh(self):
        if self.stage != self.STAGE_TUNE:
            return
        for key, label, kind, spec in self.TUNE_PAGES[self.tune_page]:
            item = self.tune_vals.get(key)
            if item:
                self.canvas.itemconfigure(item, text=self._tune_value_text(key, kind, spec))
        if self.tune_line:
            self.canvas.itemconfigure(self.tune_line, text=self._tune_values_line())

    def _tune_from_dump(self, d):
        for reg in self.TUNE_REGS:
            if reg in d:
                self.tune["regs"][reg] = d[reg]
        fmt = self.tune["regs"].get(0x24)
        for name, code, cm in self.TUNE_FORMATS:
            if code == fmt:
                self.tune["cm"] = cm
        self._tune_refresh()

    def _tune_update(self, h, frame, seq):
        if self._bars_pending and h.get("raw_seq", 0) != self._raw_seen:
            raw, self._raw_seen = self.link.take_raw()
            self._bars_pending = False
            self.link.send_bytes(b"b")                       # bars off again
            img, verdict, passed = analyze_bars(raw, 320, 240) if raw else (None, "no raw", False)
            try:
                with open(os.path.expanduser("~/endoscope_bars.txt"), "a", encoding="utf-8") as f:
                    f.write(time.strftime("%Y-%m-%d %H:%M:%S  ") + verdict + "\n")
            except Exception:
                pass
            if img is not None:
                self.diag_photo = img
                self.set_stage(self.STAGE_RUN)
                self.toast(verdict[:60], OK if passed else WARN, ms=4000)
                return
            self.toast(verdict, WARN, ms=2500)
        if h.get("regs_seq", 0) != self._tune_pop_seq:
            self._tune_pop_seq = h.get("regs_seq", 0)
            d, _ = self.link.get_regs()
            if d:
                self._tune_from_dump(d)
        ack = h.get("reg_ack")
        if ack and ack[0] != self._tune_ack_seq:
            self._tune_ack_seq = ack[0]
            if not ack[4]:
                self.toast("REG 0x%02X WRITE FAILED (read back 0x%02X)"
                           % (ack[1], ack[3]), WARN, ms=1800)
        if frame is not None and seq != self.last_seq and self.tune_video:
            self.last_seq = seq
            img = cv2.rotate(frame, cv2.ROTATE_180) if self.rot180 else frame
            fh, fw = img.shape[:2]
            bw, bh = self.tune_box[2], self.tune_box[3]
            scale = min(bw / fw, bh / fh)
            small = cv2.resize(img, (max(1, int(fw * scale)), max(1, int(fh * scale))),
                               interpolation=cv2.INTER_LINEAR)
            if self.swap_rb:
                small = small[:, :, ::-1].copy()
            small = self._pi_colour(small)
            if self.awb:
                small, self._awb_gains = grey_world(small, self._awb_gains)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            self.tune_photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.canvas.itemconfigure(self.tune_video, image=self.tune_photo)

    # ---- reset / apply / done

    def tune_reset(self):
        _, defaults = self.link.get_regs()
        if not defaults:
            self.toast("NO DEFAULTS FROM PROBE YET", WARN)
            return
        for reg in self.TUNE_REGS:
            if reg in defaults and reg != 0x24:
                self._tune_write(reg, defaults[reg])
        if 0x24 in defaults:
            self.tune["cm"] = "0" if defaults[0x24] == 0xa6 else "2"
            self._tune_write(0x24, defaults[0x24])
            self._tune_send_cm()
        self.pi_gains, self.pi_chan, self.pi_matrix = [1.0, 1.0, 1.0], [0, 1, 2], None
        self.save_cfg()
        self.toast("SENSOR + PI CORRECTION RESET TO DEFAULTS", OK, ms=1600)

    def _tune_apply_saved(self):
        saved = self.tune_saved or {}
        regs = saved.get("regs") or {}
        order = ["22", "14", "03", "04", "50", "5a", "5b", "5c", "b1", "b2",
                 "b3", "b5", "24"]
        n = 0
        for k in order:
            if k in regs:
                reg, val = int(k, 16), int(regs[k])
                self.tune["regs"][reg] = val
                self.link.send_bytes(bytes([ord("W"), reg, val]))
                n += 1
        if "cm" in saved:
            self.tune["cm"] = str(saved["cm"])
            self._tune_send_cm()
        if n:
            self.toast("TUNING RE-APPLIED TO PROBE ({} regs)".format(n), OK, ms=1600)
        self._tune_refresh()

    def tune_done(self):
        """Leave TUNE: persist everything, replay on every future probe boot."""
        self.tune_saved = {"regs": {"%02x" % r: self._reg(r, 0)
                                    for r in self.TUNE_REGS if r in self.tune["regs"]},
                           "cm": self.tune["cm"]}
        self.save_cfg()
        try:
            with open(os.path.expanduser("~/endoscope_tune.txt"), "a",
                      encoding="utf-8") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S  ") + self._tune_values_line() + "\n")
        except Exception:
            pass
        self._tune_gen_applied = self.link.health()["generation"]
        self.set_stage(self.STAGE_RUN)
        self.toast("TUNING SAVED \u2713", OK, ms=1600)


    # ---- CALIBRATION WIZARD: the Pi screen is the colour reference --------
    #
    # The operator points the probe at this screen. Each step fills the
    # screen with a known colour; the app measures what the camera reports
    # for the centre of its frame and closes the loop: sensor WB gains for
    # neutrals (best signal), Pi-side fixed gains for whatever the sensor
    # cannot reach, a channel-map check from the red/green/blue patches,
    # chroma saturation from red/blue, black level from black. Every
    # measurement is shown as numbers, so nothing here relies on eyeballing.


    def _build_lut(self):
        """
        One 256-entry curve per channel: subtract black, scale to white, then
        gamma. Precomputed because doing this per pixel on a Pi would cost
        more than the video decode.
        """
        lut = np.zeros((256, 3), np.uint8)
        x = np.arange(256, dtype=np.float32)
        for ch in range(3):
            lo = float(self.pi_black[ch])
            hi = float(self.pi_white[ch])
            if hi - lo < 8.0:                      # nonsense capture: bypass
                lo, hi = 0.0, 255.0
            n = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
            g = max(0.2, min(5.0, float(self.pi_gamma[ch])))
            if abs(g - 1.0) > 0.01:
                n = np.power(n, 1.0 / g)
            n = n * 255.0 * float(self.pi_gains[ch])
            lut[:, ch] = np.clip(n, 0, 255).astype(np.uint8)
        self._lut = lut
        return lut

    def _levels_active(self):
        return (any(b > 0.5 for b in self.pi_black)
                or any(w < 254.5 for w in self.pi_white)
                or any(abs(g - 1.0) > 0.01 for g in self.pi_gamma)
                or any(abs(g - 1.0) > 0.01 for g in self.pi_gains))

    def _pi_colour(self, img):
        """
        Channel map, then levels/gamma/gain, then the colour matrix, then
        brightness and saturation.

        Order is not arbitrary. The matrix is solved from primaries measured
        after the levels correction, so it has to be applied in that same
        space or it is describing a picture that no longer exists.
        """
        if self.pi_chan != [0, 1, 2]:
            img = img[:, :, self.pi_chan]

        if self._levels_active():
            lut = self._lut if self._lut is not None else self._build_lut()
            out = np.empty_like(img)
            for ch in range(3):
                out[:, :, ch] = lut[:, ch][img[:, :, ch]]
            img = out

        if self.pi_matrix:
            M = np.array(self.pi_matrix, np.float32)          # 4x3: B,G,R,offset
            flat = img.reshape(-1, 3).astype(np.float32)
            img = np.clip(flat @ M[:3] + M[3], 0,
                          255).astype(np.uint8).reshape(img.shape)

        need_bri = abs(self.pi_bright) > 0.5
        need_sat = abs(self.pi_sat - 1.0) > 0.01
        if not (need_bri or need_sat):
            return img
        o = img.astype(np.float32)
        if need_bri:
            o += self.pi_bright
        if need_sat:
            grey = o.mean(axis=2, keepdims=True)
            o = grey + (o - grey) * self.pi_sat
        return np.clip(o, 0, 255).astype(np.uint8)

    # ---- colour panel: manual control, plus one reliable automatic ---------

    # Rows are grouped: only one group is on screen at a time, so five large
    # touch targets never have to share the space with fifteen.
    COLOUR_GROUPS = [
        ("LEVELS", [("BLACK R", "black", 2, 2.0, 0.0, 200.0),
                    ("BLACK G", "black", 1, 2.0, 0.0, 200.0),
                    ("BLACK B", "black", 0, 2.0, 0.0, 200.0)]),
        ("WHITE",  [("WHITE R", "white", 2, 2.0, 40.0, 255.0),
                    ("WHITE G", "white", 1, 2.0, 40.0, 255.0),
                    ("WHITE B", "white", 0, 2.0, 40.0, 255.0)]),
        ("GAMMA",  [("GAMMA R", "gamma", 2, 0.05, 0.2, 5.0),
                    ("GAMMA G", "gamma", 1, 0.05, 0.2, 5.0),
                    ("GAMMA B", "gamma", 0, 0.05, 0.2, 5.0)]),
        ("IMAGE",  [("RED GAIN", "gain", 2, 0.05, 0.25, 4.0),
                    ("GREEN GAIN", "gain", 1, 0.05, 0.25, 4.0),
                    ("BLUE GAIN", "gain", 0, 0.05, 0.25, 4.0),
                    ("BRIGHTNESS", "bri", 0, 4.0, -80.0, 80.0),
                    ("SATURATION", "sat", 0, 0.05, 0.0, 2.0)]),
    ]

    def _colour_value(self, kind, idx):
        return {"gain": self.pi_gains, "black": self.pi_black,
                "white": self.pi_white, "gamma": self.pi_gamma}.get(
            kind, [self.pi_bright if kind == "bri" else self.pi_sat])[
            idx if kind in ("gain", "black", "white", "gamma") else 0]

    def _colour_set(self, kind, idx, val, lo, hi):
        val = max(lo, min(hi, val))
        if kind == "gain":
            self.pi_gains[idx] = round(val, 3)
        elif kind == "black":
            self.pi_black[idx] = round(val, 1)
        elif kind == "white":
            self.pi_white[idx] = round(val, 1)
        elif kind == "gamma":
            self.pi_gamma[idx] = round(val, 3)
        elif kind == "bri":
            self.pi_bright = round(val, 1)
        else:
            self.pi_sat = round(val, 3)
        self._lut = None                       # curve changed, rebuild it
        self.last_seq = -1                     # force the preview to redraw
        self._refresh_colour_readouts()

    def enter_cal(self):
        """
        Anchor black, white and mid, then fine-tune by hand.

        The old wizard asked the operator to aim at coloured patches on this
        screen and inferred a matrix. Too much rode on things it could not
        see -- aim, screen brightness, viewing angle, reflections -- and one
        bad patch poisoned the result, which is why every automatic answer
        came out wrong.

        These three captures ask for something unambiguous instead: cover the
        lens, show white paper, show grey. Each fixes one end of the transfer
        curve by arithmetic. Anchoring only white, as a gain does, cannot fix
        this sensor, whose blue is close at the highlights and a third of the
        others in the midtones.
        """
        c, W, H, z = self.canvas, self.W, self.H, self.stage_items
        z.append(c.create_rectangle(0, 0, W, H, fill="#0b1114", outline=""))

        pad = max(10, int(H / 44))
        title = max(13, int(H / 26))
        row_f = max(10, int(H / 34))
        small = max(9, int(H / 48))

        z.append(c.create_text(pad, pad, anchor="nw", text="COLOUR",
                               fill="#ffffff", font=self.f(title, True)))

        pv_w = int(W * 0.38)
        pv_h = int(pv_w * 0.75)
        pv_x = W - pv_w - pad
        pv_y = pad + int(title * 2.6)
        z.append(c.create_rectangle(pv_x - 2, pv_y - 2, pv_x + pv_w + 2,
                                    pv_y + pv_h + 2, outline=EDGE, width=2))
        self.cal_thumb = c.create_image(pv_x + pv_w / 2, pv_y + pv_h / 2)
        z.append(self.cal_thumb)
        self.cal_thumb_dims = (pv_w, pv_h)

        rs = int(min(pv_w, pv_h) * 0.20)
        z.append(c.create_rectangle(pv_x + pv_w / 2 - rs, pv_y + pv_h / 2 - rs,
                                    pv_x + pv_w / 2 + rs, pv_y + pv_h / 2 + rs,
                                    outline="#9ccc65", width=2, dash=(5, 4)))
        self.cal_text = c.create_text(pv_x + pv_w / 2, pv_y + pv_h + small * 1.8,
                                      fill="#9ccc65", font=self.f(small, True),
                                      text="")
        z.append(self.cal_text)
        self.cal_hint = c.create_text(pv_x + pv_w / 2, pv_y + pv_h + small * 3.4,
                                      fill=DIM, font=self.f(small),
                                      text="fill the dashed square with the target")
        z.append(self.cal_hint)

        col_w = pv_x - pad * 2
        btn = max(int(H * 0.085), 38)

        # Group selector: one tap swaps which three controls are on screen.
        gy = pv_y
        gw = int((col_w - 3 * 6) / 4)
        for i, (name, _) in enumerate(self.COLOUR_GROUPS):
            sel = i == self.cal_group
            self._pill(pad + i * (gw + 6), gy, int(btn * 0.8), name,
                       "#00695c" if sel else "#2b3b44",
                       lambda ix=i: self._set_group(ix), z, width=gw)

        rows = self.COLOUR_GROUPS[self.cal_group][1]
        top = gy + int(btn * 0.8) + pad
        avail = (H - pad - btn * 2 - int(pad * 1.6)) - top
        row_h = int(avail / max(len(rows), 1))
        rb = min(btn, max(34, int(row_h * 0.72)))
        self.colour_readouts = {}
        for i, (label, kind, idx, step, lo, hi) in enumerate(rows):
            y = top + i * row_h
            self._pill(pad, y, rb, "\u2212", "#37474f",
                       lambda k=kind, ix=idx, st=step, l=lo, h2=hi:
                       self._colour_set(k, ix, self._colour_value(k, ix) - st, l, h2), z)
            self._pill(pad + col_w - rb, y, rb, "+", "#37474f",
                       lambda k=kind, ix=idx, st=step, l=lo, h2=hi:
                       self._colour_set(k, ix, self._colour_value(k, ix) + st, l, h2), z)
            z.append(c.create_text(pad + rb + 10, y + rb / 2, anchor="w",
                                   text=label, fill="#eceff1",
                                   font=self.f(row_f, True)))
            r = c.create_text(pad + col_w - rb - 10, y + rb / 2, anchor="e",
                              text="", fill="#80cbc4", font=self.f(row_f, True))
            z.append(r)
            self.colour_readouts[label] = (r, kind, idx)

        bs = max(10, int(H / 44))
        row2 = H - pad - btn
        row1 = row2 - btn - int(pad * 0.6)

        def bar(y, items):
            widths = [self.text_w(t, bs, True) + 28 for t, _, _ in items]
            gap = max(6, int((W - pad * 2 - sum(widths)) / max(len(items) - 1, 1)))
            x = pad
            for (label, col, cb), w in zip(items, widths):
                self._pill(x, y, btn, label, col, cb, z, width=w)
                x += w + gap

        # Neutrals first: these alone fix a cast, and are all most sensors need.
        bar(row1, [("BLACK", "#263238", self.cal_set_black),
                   ("WHITE", "#546e7a", self.cal_set_white),
                   ("GREY", "#00695c", self.cal_set_grey),
                   ("RESET", "#5d4037", self.cal_reset),
                   ("SAVE \u2713", "#2e7d32", self.cal_save)])
        # Primaries second: only for hue errors that survive the neutrals.
        bar(row2, [("RED", "#b71c1c", lambda: self.cal_set_primary("R")),
                   ("GREEN", "#1b5e20", lambda: self.cal_set_primary("G")),
                   ("BLUE", "#0d47a1", lambda: self.cal_set_primary("B")),
                   ("NO MATRIX", "#37474f", self.cal_clear_matrix)])

        self.button(W - pad, pad, "\u2715", "#c62828",
                    lambda: self.set_stage(self.STAGE_TUNE), bs,
                    store=z, anchor="ne")

        self.cal_meas = None
        self.last_seq = -1
        self._refresh_colour_readouts()

    def _set_group(self, i):
        self.cal_group = i
        self.set_stage(self.STAGE_CAL)

    def _pill(self, x, y, h, label, fill, cb, store, width=None):
        w = width if width else h
        r = self.canvas.create_rectangle(x, y, x + w, y + h, fill=fill,
                                         outline="", width=0)
        t = self.canvas.create_text(x + w / 2, y + h / 2, text=label,
                                    fill="white",
                                    font=self.f(max(11, int(h * 0.34)), True))
        for item in (r, t):
            self.canvas.tag_bind(item, "<Button-1>", lambda e: cb())
        store.extend([r, t])
        return r, t

    def _refresh_colour_readouts(self):
        if self.stage != self.STAGE_CAL:
            return
        for label, (item, kind, idx) in self.colour_readouts.items():
            v = self._colour_value(kind, idx)
            if kind in ("black", "white"):
                txt = "{:.0f}".format(v)
            elif kind == "bri":
                txt = "{:+.0f}".format(v)
            else:
                txt = "{:.2f}".format(v)
            try:
                self.canvas.itemconfigure(item, text=txt)
            except Exception:
                pass

    def _cal_patch(self, what):
        """The mean B,G,R of the sampled square, before any correction."""
        if self.cal_meas is None:
            self.toast("NO PICTURE YET", WARN)
            return None
        raw = self.cal_meas[0]
        if what == "black":
            # Brightness is the wrong test here. Whatever the sensor still
            # reads with the lens covered IS the black level -- on this GC0308
            # that is a green-tinted value well above zero, and rejecting it
            # for being too bright threw away the very measurement wanted.
            # A covered lens is FLAT; a real scene has texture. Judge that.
            flat = self.cal_meas[2] if len(self.cal_meas) > 2 else None
            if flat is not None and flat > 14.0:
                self.toast("LENS NOT COVERED \u2014 the square still has "
                           "detail in it", WARN, ms=2400)
                return None
        flat = self.cal_meas[2] if len(self.cal_meas) > 2 else 0.0

        if what != "black":
            if float(np.mean(raw)) < 12:
                self.toast("TOO DARK \u2014 more light on the card", WARN, ms=2200)
                return None
            if flat > 20.0:
                # A card fills the square evenly. Texture in it means the
                # square is seeing something else as well, and the average
                # would then describe neither.
                self.toast("SQUARE NOT FILLED \u2014 move closer so the card "
                           "covers it", WARN, ms=2600)
                return None
        if what == "white" and float(np.max(raw)) > 250:
            # A clipped channel has lost its true value, so it cannot define
            # the top of the curve.
            self.toast("OVEREXPOSED \u2014 less light, or tilt the card away",
                       WARN, ms=2400)
            return None
        if what == "grey":
            lo = np.array(self.pi_black, np.float32)
            hi = np.array(self.pi_white, np.float32)
            n = np.clip((raw - lo) / np.maximum(hi - lo, 1e-3), 0, 1)
            if float(np.mean(n)) < 0.12 or float(np.mean(n)) > 0.88:
                # Solving gamma from a near-black or near-white patch gives a
                # wild exponent from almost no signal.
                self.toast("NOT MID GREY \u2014 it reads {:.0f}% of the way "
                           "from black to white".format(float(np.mean(n)) * 100),
                           WARN, ms=2800)
                return None
        return raw

    def cal_set_black(self):
        """Cover the lens. Whatever it still reads is the floor, per channel."""
        raw = self._cal_patch("black")
        if raw is None:
            return
        self.pi_black = [round(float(v), 1) for v in raw]
        self._after_capture("BLACK POINT SET  R{:.0f} G{:.0f} B{:.0f}".format(
            raw[2], raw[1], raw[0]))

    def cal_set_white(self):
        """Show white paper. That reading becomes 255 on every channel, which
        balances the highlights and sets the top of the curve in one step."""
        raw = self._cal_patch("white")
        if raw is None:
            return
        self.pi_white = [round(max(float(v), self.pi_black[i] + 12.0), 1)
                         for i, v in enumerate(raw)]
        self._after_capture("WHITE POINT SET  R{:.0f} G{:.0f} B{:.0f}".format(
            raw[2], raw[1], raw[0]))

    def cal_set_grey(self):
        """
        Show mid grey. Black and white pin the ends; this pins the middle.

        Without it a channel can match at both ends and still be wrong
        everywhere between, which is this sensor's actual fault: blue is 22
        low at the highlights and 100 low at the median.
        """
        raw = self._cal_patch("grey")
        if raw is None:
            return
        gam = []
        for ch in range(3):
            lo, hi = self.pi_black[ch], self.pi_white[ch]
            if hi - lo < 8:
                gam.append(1.0)
                continue
            n = float(np.clip((raw[ch] - lo) / (hi - lo), 0.02, 0.98))
            # Solve n**(1/g) = 0.5 so the grey card lands at half scale.
            gam.append(round(float(np.clip(math.log(n) / math.log(0.5),
                                           0.2, 5.0)), 3))
        self.pi_gamma = gam
        self._after_capture("MID SET  gamma R{:.2f} G{:.2f} B{:.2f}".format(
            gam[2], gam[1], gam[0]))

    def cal_set_primary(self, which):
        """
        Capture one primary. Three of them define a colour matrix.

        Black, white and grey fix each channel's own curve, which is enough
        whenever the fault is a cast. They cannot fix cross-talk -- red light
        landing partly in the green channel -- because that is a relationship
        between channels, not a level within one. If neutrals come out neutral
        and hues are still wrong, that is the remaining fault, and only a
        matrix addresses it.
        """
        raw = self._cal_patch("primary")
        if raw is None:
            return
        # Measured in the levels-corrected space, which is where the matrix
        # will be applied.
        corrected = self._pi_colour(
            np.full((4, 4, 3), raw, dtype=np.uint8)).reshape(-1, 3).mean(axis=0)
        self.cal_prim[which] = [float(v) for v in corrected]
        have = "".join(k for k in "RGB" if k in self.cal_prim)
        self.toast("{} CAPTURED  ({} of RGB done)".format(which, len(have)),
                   OK, ms=2000)
        if len(self.cal_prim) == 3:
            self._solve_matrix()

    def _solve_matrix(self):
        """
        Solve M so the three measured primaries map onto pure red, green and
        blue. Refused if the measurements are nearly parallel, because
        inverting that produces enormous coefficients and a garish picture.
        """
        try:
            P = np.array([self.cal_prim["B"], self.cal_prim["G"],
                          self.cal_prim["R"]], np.float32).T      # BGR columns
            T = np.eye(3, dtype=np.float32) * 255.0
            if abs(float(np.linalg.det(P))) < 1e3:
                self.toast("PRIMARIES TOO ALIKE \u2014 recapture with stronger "
                           "colours", WARN, ms=2800)
                return
            M = (T @ np.linalg.inv(P)).T
            if float(np.abs(M).max()) > 6.0:
                self.toast("MATRIX UNSTABLE \u2014 ignoring it", WARN, ms=2600)
                return
            self.pi_matrix = [[float(v) for v in row] for row in M] + [[0.0, 0.0, 0.0]]
            self._after_capture("COLOUR MATRIX SET from R, G and B")
        except Exception as e:
            self.toast("MATRIX FAILED: {}".format(e), WARN, ms=2600)

    def cal_clear_matrix(self):
        self.pi_matrix = None
        self.cal_prim = {}
        self.last_seq = -1
        self.toast("COLOUR MATRIX CLEARED \u2014 neutrals only", OK, ms=1800)

    def _after_capture(self, msg):
        self.pi_matrix = None                  # a matrix would fight the curve
        self._lut = None
        self.last_seq = -1
        self._refresh_colour_readouts()
        self.toast(msg, OK, ms=2400)

    def cal_reset(self):
        self.pi_black = [0.0, 0.0, 0.0]
        self.pi_white = [255.0, 255.0, 255.0]
        self.pi_gamma = [1.0, 1.0, 1.0]
        self._lut = None
        self.pi_gains = [1.0, 1.0, 1.0]
        self.pi_chan = [0, 1, 2]
        self.pi_matrix = None
        self.pi_bright = 0.0
        self.pi_sat = 1.0
        self.last_seq = -1
        self._refresh_colour_readouts()
        self.toast("COLOUR RESET", OK, ms=1400)

    def _cal_measure(self, frame):
        """Mean B,G,R of the centre 40% of the decoded frame: raw and after
        the Pi correction."""
        fh, fw = frame.shape[:2]
        roi = frame[int(fh * 0.3):int(fh * 0.7), int(fw * 0.3):int(fw * 0.7)]
        raw = roi.reshape(-1, 3).mean(axis=0)
        cor = self._pi_colour(roi).reshape(-1, 3).mean(axis=0)
        # Spatial spread of the patch: near zero when the lens is covered or
        # aimed at a card, large when it is looking at a scene.
        flat = float(roi.reshape(-1, 3).mean(axis=1).std())
        return raw, cor, flat

    def _cal_update(self, h, frame, seq):
        """Refresh the preview and the sampled-square numbers."""
        if frame is None or seq == self.last_seq:
            return
        self.last_seq = seq
        img = cv2.rotate(frame, cv2.ROTATE_180) if self.rot180 else frame
        raw, cor, flat = self._cal_measure(img)
        self.cal_meas = (raw, cor, flat)

        if self.cal_text:
            # Only what the operator can act on: the square's colour now, and
            # how far off neutral it still is.
            spread = float(np.max(cor) - np.min(cor))
            self.canvas.itemconfigure(
                self.cal_text,
                text="square  R {:3.0f}  G {:3.0f}  B {:3.0f}     off-neutral {:3.0f}"
                     .format(cor[2], cor[1], cor[0], spread))

        tw, th = self.cal_thumb_dims
        small = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        small = self._pi_colour(small)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        self.cal_thumb_photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        if self.cal_thumb:
            self.canvas.itemconfigure(self.cal_thumb, image=self.cal_thumb_photo)

    def _cal_channel_check(self):
        """After RED/GREEN/BLUE: which camera channel answered each patch."""
        perm = [0, 1, 2]
        for idx, name in ((2, "RED"), (1, "GREEN"), (0, "BLUE")):
            m = self.cal_perm.get(name)
            if m is not None:
                perm[idx] = int(np.argmax(m))
        return perm

    def cal_save(self):
        # Colour correction matrix from the measured patches (least squares,
        # camera raw -> displayed colour, with offset). This is the standard
        # camera-profile approach and does not depend on any sensor register
        # behaving; it needs unclipped, well-exposed measurements.
        targets = {"WHITE": (235, 235, 235), "RED": (0, 0, 235), "GREEN": (0, 235, 0),
                   "BLUE": (235, 0, 0), "BLACK": (8, 8, 8)}
        have = [k for k in targets if k in self.cal_perm]
        if len(have) >= 4:
            X = np.array([list(self.cal_perm[k]) + [1.0] for k in have], np.float64)
            T = np.array([targets[k] for k in have], np.float64)
            M, _, rank, _ = np.linalg.lstsq(X, T, rcond=None)
            if rank == 4 and np.all(np.abs(M[:3]) < 6.0) and np.all(np.abs(M[3]) < 200):
                self.pi_matrix = [[float(v) for v in row] for row in M]
                self.pi_gains = [1.0, 1.0, 1.0]
                self.toast("COLOUR MATRIX FITTED FROM {} PATCHES \u2713".format(len(have)),
                           OK, ms=1800)
        if all(k in self.cal_perm for k in ("RED", "GREEN", "BLUE")):
            perm = self._cal_channel_check()
            if sorted(perm) == [0, 1, 2]:
                self.pi_chan = perm
                if perm != [0, 1, 2]:
                    self.toast("CHANNEL MAP CORRECTED: BGR -> {}".format(perm), OK, ms=1800)
        self.tune_saved = {"regs": {"%02x" % r: self._reg(r, 0)
                                    for r in self.TUNE_REGS if r in self.tune["regs"]},
                           "cm": self.tune["cm"]}
        self.save_cfg()
        try:
            with open(os.path.expanduser("~/endoscope_tune.txt"), "a", encoding="utf-8") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S  CAL ") + self._tune_values_line()
                        + "  pi_gains={} chan={}\n".format(self.pi_gains, self.pi_chan))
        except Exception:
            pass
        self.cal_perm = {}
        self.set_stage(self.STAGE_RUN)
        self.toast("CALIBRATION SAVED \u2713", OK, ms=1800)

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
    ap.add_argument("--video", metavar="N",
                    help="take video from a standard UVC camera instead of the "
                         "serial stream, e.g. --video auto or --video /dev/video0. The "
                         "probe then supplies attitude only, and none of the "
                         "firmware colour handling is in the picture path.")
    ap.add_argument("--video-size", default="640x480",
                    help="requested UVC capture size (default 640x480)")
    ap.add_argument("--official", action="store_true",
                    help="use M5Stack stock firmware: UVC video plus raw IMU from "
                         "ws://192.168.4.1; no custom colour path")
    ap.add_argument("--imu-ws", default="ws://192.168.4.1/api/v1/ws/imu_data",
                    help="stock-firmware IMU WebSocket URL")
    ap.add_argument("--legacy-colour-tools", action="store_true",
                    help="show the old COLOR/DIAG/TUNE controls (diagnostics only)")
    ap.add_argument("--windowed", action="store_true")
    ap.add_argument("--sim", action="store_true",
                    help="synthetic probe: run the whole UI with no hardware")
    ap.add_argument("--log", metavar="FILE.csv",
                    help="record time, az, el, quaternion for drift analysis")
    args = ap.parse_args()

    try:
        vw, vh = (int(x) for x in args.video_size.lower().split("x", 1))
        if vw < 16 or vh < 16:
            raise ValueError
    except ValueError:
        ap.error("--video-size must look like 640x480")

    if args.sim:
        link = SimLink().start()
    elif args.official:
        link = OfficialProbeLink(args.video or "auto", vw, vh,
                                 args.imu_ws).start()
    else:
        link = ProbeLink(args.port, args.baud)
        if args.video:
            link.camera = V4L2Source(args.video, vw, vh).start()
        link.start()
    app = App(link, args)
    try:
        app.run()
    finally:
        if (getattr(link, 'camera', None) and
                not getattr(link, 'is_uvc', False)):
            link.camera.stop()
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
