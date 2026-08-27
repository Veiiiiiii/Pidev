#!/usr/bin/env python3
"""
Raspberry Pi Fullscreen Camera  (v2 - colour fix)
-------------------------------------------------
Click the icon -> camera fills the screen. Tap anywhere to exit.

Exit with the red EXIT button on the right-hand edge. Tapping the picture
does nothing, so you cannot close it by accident.

If the colours ever look wrong (new camera, etc.) press SPACE to step through
the pixel-format options. The working one is saved to ~/.config/picam.conf and
reused on every later launch.

Run `python3 cam.py --check` for a diagnostic report.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time

CONFIG = os.path.join(os.path.expanduser("~"), ".config", "picam.conf")


# --------------------------------------------------------------- profiles
# Each profile = one way of pulling and interpreting pixels.
# (label, fourcc or None, let OpenCV convert?, conversion code or None)
PROFILES = [
    ("MJPG standard",   "MJPG", True,  "BGR2RGB"),
    ("YUYV standard",   "YUYV", True,  "BGR2RGB"),
    ("YUYV raw",        "YUYV", False, "YUYV"),
    ("UYVY raw",        "YUYV", False, "UYVY"),
    ("MJPG no swap",    "MJPG", True,  None),
    ("Default no swap",  None,  True,  None),
    ("Default standard", None,  True,  "BGR2RGB"),
    ("I420 raw",        "YU12", False, "I420"),
    ("Greyscale",        None,  True,  "GRAY"),
]


def load_profile_index():
    try:
        with open(CONFIG, encoding="utf-8") as f:
            idx = int(json.load(f).get("profile", 0))
        return idx if 0 <= idx < len(PROFILES) else 0
    except Exception:
        return 0


def save_profile_index(idx):
    try:
        os.makedirs(os.path.dirname(CONFIG), exist_ok=True)
        with open(CONFIG, "w", encoding="utf-8") as f:
            json.dump({"profile": idx}, f)
    except Exception:
        pass


# --------------------------------------------------------------- discovery

def video_devices():
    def num(path):
        digits = "".join(c for c in os.path.basename(path) if c.isdigit())
        return int(digits) if digits else 0
    return sorted(glob.glob("/dev/video*"), key=num)


def device_index(path):
    digits = "".join(c for c in os.path.basename(path) if c.isdigit())
    return int(digits) if digits else 0


def find_camera(verbose=False):
    """A Pi 5 exposes many /dev/video* nodes that are codecs, not cameras."""
    devices = video_devices()
    if not devices:
        return None
    try:
        import cv2
    except ImportError:
        return devices[0]

    for dev in devices:
        cap = cv2.VideoCapture(device_index(dev))
        opened = cap.isOpened()
        ok = False
        if opened:
            ok, _ = cap.read()
        cap.release()
        if verbose:
            print(f"  {dev}: opened={opened} frame={ok}")
        if ok:
            return dev
    return None


def show_message(text):
    try:
        import tkinter as tk
        root = tk.Tk()
        root.attributes("-fullscreen", True)
        root.configure(bg="#111111")
        tk.Label(root, text=text, fg="white", bg="#111111",
                 font=("DejaVu Sans", 18), justify="left",
                 wraplength=root.winfo_screenwidth() - 140).pack(expand=True)
        tk.Label(root, text="Tap the screen to close", fg="#888888",
                 bg="#111111", font=("DejaVu Sans", 14)).pack(pady=40)
        root.bind("<Button-1>", lambda e: root.destroy())
        root.bind("<Escape>", lambda e: root.destroy())
        root.mainloop()
    except Exception:
        print(text, file=sys.stderr)


# --------------------------------------------------------------- capture

def open_capture(cv2, index, profile):
    """Open the camera configured for one profile. Returns the capture or None."""
    _, fourcc, convert_rgb, _ = profile

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None

    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    # CAP_PROP_CONVERT_RGB=0 hands back the sensor's raw bytes so we can
    # decode them ourselves; =1 lets OpenCV do its own (sometimes wrong) guess.
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 1 if convert_rgb else 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None
    return cap


def to_rgb(cv2, frame, conversion):
    """Turn whatever the camera gave us into a normal RGB array."""
    import numpy as np

    if frame is None:
        return None

    try:
        if conversion == "BGR2RGB":
            if frame.ndim == 3 and frame.shape[2] == 3:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame.ndim == 2:
                return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            return None
        if conversion == "YUYV":
            return cv2.cvtColor(frame, cv2.COLOR_YUV2RGB_YUYV)
        if conversion == "UYVY":
            return cv2.cvtColor(frame, cv2.COLOR_YUV2RGB_UYVY)
        if conversion == "I420":
            return cv2.cvtColor(frame, cv2.COLOR_YUV2RGB_I420)
        if conversion == "GRAY":
            if frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        # conversion is None -> already usable, just make sure it is 3-channel
        if frame.ndim == 3 and frame.shape[2] == 3:
            return frame
        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        return None
    except cv2.error:
        return None
    except Exception:
        return None


# --------------------------------------------------------------- backend A

def run_opencv(device):
    import cv2
    import tkinter as tk
    from PIL import Image, ImageTk

    index = device_index(device)
    state = {"profile": load_profile_index(), "cap": None, "running": True}

    root = tk.Tk()
    root.title("Camera")
    root.configure(bg="black", cursor="none")
    root.attributes("-fullscreen", True)
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()

    view = tk.Label(root, bg="black", borderwidth=0, highlightthickness=0)
    view.place(x=0, y=0, relwidth=1, relheight=1)

    # --- EXIT button, right-hand side, sized for a fingertip ---
    exit_btn = tk.Label(root, text="\u2715\nEXIT", fg="white", bg="#d32f2f",
                        font=("DejaVu Sans", 22, "bold"), padx=26, pady=22)
    exit_btn.place(relx=0.965, rely=0.5, anchor="e")

    status = tk.Label(root, text="", fg="white", bg="#000000",
                      font=("DejaVu Sans", 14))
    status.place(relx=0.5, rely=0.055, anchor="center")

    def flash_status(text, ms=2500):
        status.configure(text=text)
        status.after(ms, lambda: status.configure(text=""))

    def quit_app(event=None):
        state["running"] = False
        if state["cap"] is not None:
            try:
                state["cap"].release()
            except Exception:
                pass
        try:
            root.destroy()
        except Exception:
            pass

    def apply_profile(idx, announce=True):
        """Reopen the camera using profile idx."""
        if state["cap"] is not None:
            try:
                state["cap"].release()
            except Exception:
                pass
            state["cap"] = None
        profile = PROFILES[idx]
        cap = open_capture(cv2, index, profile)
        state["cap"] = cap
        state["profile"] = idx
        save_profile_index(idx)
        if announce:
            label = profile[0]
            ok = "" if cap else "  (this one failed)"
            flash_status(f"[{idx + 1}/{len(PROFILES)}]  {label}{ok}")

    def next_profile(event=None):
        apply_profile((state["profile"] + 1) % len(PROFILES))
        return "break"   # stop the click from also triggering exit

    exit_btn.bind("<Button-1>", quit_app)
    root.bind("<Escape>", quit_app)
    root.bind("<q>", quit_app)
    # Format cycling is still available from the keyboard (space) in case a
    # different camera is plugged in later, but it has no on-screen button.
    root.bind("<space>", next_profile)
    root.protocol("WM_DELETE_WINDOW", quit_app)

    apply_profile(state["profile"], announce=False)
    if state["cap"] is None:
        # saved profile no longer works -> fall back to the first one
        apply_profile(0, announce=False)
    if state["cap"] is None:
        root.destroy()
        return False

    exit_btn.lift()

    def update():
        if not state["running"]:
            return
        cap = state["cap"]
        if cap is not None:
            ok, frame = cap.read()
            if ok:
                rgb = to_rgb(cv2, frame, PROFILES[state["profile"]][3])
                if rgb is not None:
                    h, w = rgb.shape[:2]
                    scale = min(screen_w / w, screen_h / h)
                    rgb = cv2.resize(rgb, (max(int(w * scale), 1),
                                           max(int(h * scale), 1)),
                                     interpolation=cv2.INTER_LINEAR)
                    photo = ImageTk.PhotoImage(Image.fromarray(rgb))
                    view.configure(image=photo)
                    view.image = photo   # keep a reference or it goes black
                    exit_btn.lift()      # stay above the video
        root.after(15, update)

    root.after(50, update)
    root.mainloop()
    return True


# --------------------------------------------------------------- backend B

def run_vlc(device):
    vlc = shutil.which("vlc") or shutil.which("cvlc")
    if not vlc:
        return False

    proc = subprocess.Popen(
        [vlc, f"v4l2://{device}",
         ":v4l2-width=1280", ":v4l2-height=720",
         "--fullscreen", "--no-audio", "--no-osd",
         "--no-video-title-show", "--video-on-top",
         "--qt-notification=0"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(2.5)
    if proc.poll() is not None:
        return False

    try:
        import tkinter as tk
    except ImportError:
        proc.wait()
        return True

    win = tk.Tk()
    win.overrideredirect(True)
    win.attributes("-topmost", True)
    win.geometry(f"110x110+{win.winfo_screenwidth() - 130}+20")
    win.configure(bg="#d32f2f")

    def quit_app(event=None):
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
        try:
            win.destroy()
        except Exception:
            pass

    label = tk.Label(win, text="\u2715\nEXIT", fg="white", bg="#d32f2f",
                     font=("DejaVu Sans", 18, "bold"))
    label.pack(expand=True, fill="both")
    for widget in (win, label):
        widget.bind("<Button-1>", quit_app)
    win.bind("<Escape>", quit_app)

    def watch():
        if proc.poll() is not None:
            try:
                win.destroy()
            except Exception:
                pass
            return
        win.attributes("-topmost", True)
        win.after(1000, watch)

    win.after(1000, watch)
    win.mainloop()
    quit_app()
    return True


# --------------------------------------------------------------- backend C

PAGE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
 html,body{margin:0;height:100%;background:#000;overflow:hidden}
 video{width:100vw;height:100vh;object-fit:contain}
 #hint{position:fixed;bottom:6%;left:0;right:0;text-align:center;color:#fff;
       font:16px sans-serif;text-shadow:0 0 6px #000}
 #err{color:#fff;font:20px sans-serif;padding:40px;text-align:center}
</style></head><body>
<video id="v" autoplay playsinline muted></video>
<div id="hint">Tap anywhere to exit</div>
<script>
 setTimeout(function(){var h=document.getElementById('hint');if(h)h.remove();},4000);
 navigator.mediaDevices.getUserMedia({video:{width:1280,height:720}})
  .then(function(s){document.getElementById('v').srcObject=s;})
  .catch(function(e){document.body.innerHTML='<div id="err">Camera error: '+e.name+'</div>';});
 document.body.addEventListener('click',function(){window.close();});
</script></body></html>"""


def run_chromium():
    browser = shutil.which("chromium-browser") or shutil.which("chromium")
    if not browser:
        return False
    page = os.path.join(os.path.expanduser("~"), ".cam_fullscreen.html")
    with open(page, "w", encoding="utf-8") as f:
        f.write(PAGE)
    subprocess.run([
        browser, "--kiosk", "--start-fullscreen",
        "--use-fake-ui-for-media-stream",
        "--noerrdialogs", "--disable-infobars",
        "--disable-session-crashed-bubble",
        "--user-data-dir=" + os.path.expanduser("~/.cam_chrome_profile"),
        "file://" + page,
    ])
    return True


# --------------------------------------------------------------- diagnostics

def check():
    print("=" * 52)
    print("Raspberry Pi Camera - diagnostic report")
    print("=" * 52)
    print(f"\nPython: {sys.version.split()[0]}")
    print("Display: " + (os.environ.get("WAYLAND_DISPLAY")
                         or os.environ.get("DISPLAY") or "NONE"))
    print(f"Saved colour profile: {load_profile_index()} "
          f"({PROFILES[load_profile_index()][0]})")

    print("\nVideo devices:")
    devs = video_devices()
    print("  " + ("\n  ".join(devs) if devs else "NONE"))

    print("\nPython modules:")
    for mod in ("cv2", "numpy", "PIL", "tkinter"):
        try:
            __import__(mod)
            print(f"  {mod:10s} OK")
        except ImportError:
            print(f"  {mod:10s} MISSING")

    print("\nExternal programs:")
    for prog in ("vlc", "chromium-browser", "chromium", "v4l2-ctl"):
        print(f"  {prog:18s} {shutil.which(prog) or 'MISSING'}")

    print("\nProbing for a live signal:")
    cam = find_camera(verbose=True)
    print(f"\nUsable camera: {cam or 'NONE'}")

    if cam:
        try:
            out = subprocess.run(["v4l2-ctl", "--list-formats-ext",
                                  "-d", cam], capture_output=True, text=True)
            if out.stdout:
                print("\nSupported formats:")
                print(out.stdout)
        except FileNotFoundError:
            print("\n(v4l2-ctl not installed, cannot list formats)")
    print("=" * 52)


# --------------------------------------------------------------- main

def pick_backend():
    """Allow --backend=opencv|chromium|vlc to override the automatic order."""
    for arg in sys.argv[1:]:
        if arg.startswith("--backend="):
            return arg.split("=", 1)[1].strip().lower()
    return None


def main():
    if "--check" in sys.argv:
        check()
        return 0
    if "--reset" in sys.argv:
        save_profile_index(0)
        print("Colour profile reset to 1.")
        return 0

    device = find_camera()
    if device is None:
        show_message("No working camera found.\n\n"
                     "- Check the USB camera is plugged in\n"
                     "- Try a different USB port\n"
                     "- Run: python3 cam.py --check")
        return 1

    forced = pick_backend()

    def try_opencv():
        try:
            import cv2   # noqa: F401
            import PIL   # noqa: F401
        except ImportError:
            return False
        try:
            return run_opencv(device)
        except Exception as e:
            print(f"OpenCV backend failed: {e}", file=sys.stderr)
            return False

    def try_chromium():
        try:
            return run_chromium()
        except Exception as e:
            print(f"Chromium backend failed: {e}", file=sys.stderr)
            return False

    def try_vlc():
        try:
            return run_vlc(device)
        except Exception as e:
            print(f"VLC backend failed: {e}", file=sys.stderr)
            return False

    backends = {"opencv": try_opencv, "chromium": try_chromium, "vlc": try_vlc}

    if forced:
        fn = backends.get(forced)
        if fn is None:
            print(f"Unknown backend '{forced}'. "
                  "Use opencv, chromium or vlc.", file=sys.stderr)
            return 1
        return 0 if fn() else 1

    # Chromium sits ahead of VLC: VLC frequently mis-decodes the pixel format
    # on cheap UVC cameras and produces a green/purple picture.
    for fn in (try_opencv, try_chromium, try_vlc):
        if fn():
            return 0

    show_message("Camera detected, but no display backend available.\n\n"
                 "  sudo apt install python3-opencv python3-pil.imagetk\n"
                 "  sudo apt install vlc")
    return 1


if __name__ == "__main__":
    sys.exit(main())
