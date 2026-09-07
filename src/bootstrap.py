import os
import subprocess
import sys

DEFAULT_ROOT = "/content/macura-backflip"


# Drive
def mount_drive():
    try:
        from google.colab import drive
    except ImportError:
        print("Not on Colab - skipping Drive mount.")
        return False
    if not os.path.isdir("/content/drive/MyDrive"):
        drive.mount("/content/drive")
    print("Drive mounted.")
    return True


# Dependencies
def install(root: str = DEFAULT_ROOT):
    """Install requirements.txt (mujoco, gymnasium, stable-baselines3, ...)."""
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                    os.path.join(root, "requirements.txt")], check=True)
    print("Requirements installed.")


# Headless rendering backend (Colab has no display).
# We use OSMesa (software rendering): it renders headless and, being CPU-only, is
# CUDA-safe - unlike egl, which segfaults next to a live CUDA context. Because OSMesa
# cannot segfault, we do NOT need the slow isolated-subprocess probe (that added
# ~15-30s to setup, importing torch + CUDA + rendering just to check). Instead we
# check for libOSMesa with ctypes.util.find_library - instant - and apt-install it
# only if it is missing. If libOSMesa is present, `import mujoco` with MUJOCO_GL=osmesa
# is guaranteed to load, so no render probe is needed.
def _has_lib(name: str) -> bool:
    import ctypes.util
    try:
        return ctypes.util.find_library(name) is not None
    except Exception:
        return False


def ensure_render_backend(root: str = DEFAULT_ROOT) -> str:
    """Set MUJOCO_GL to a CUDA-safe render backend so headless video works AND training
    does not crash. Fast: no subprocess, no torch/CUDA/render probe.

    Colab -> OSMesa (software, CUDA-safe): apt-install libOSMesa only if missing, then
    use it. Local (mac) -> glfw. If OSMesa is unavailable we pin `disable` - `import
    mujoco` still succeeds, no GL context is created, and the visual cells skip
    gracefully. Honors an explicit MUJOCO_GL if the user set one.
    """
    if os.environ.get("MUJOCO_GL"):
        return os.environ["MUJOCO_GL"]

    if os.path.isdir("/content"):                       # Colab
        if not _has_lib("OSMesa"):                      # install only when needed (one-time)
            try:
                subprocess.run(["apt-get", "install", "-y", "-qq", "libosmesa6"],
                               check=False, capture_output=True, timeout=300)
            except Exception:
                pass
        if _has_lib("OSMesa"):
            os.environ["MUJOCO_GL"] = "osmesa"
            print("Render backend: osmesa (software, CUDA-safe).")
            return "osmesa"
    else:                                               # local mac/linux with a display
        os.environ["MUJOCO_GL"] = "glfw"
        print("Render backend: glfw.")
        return "glfw"

    # OSMesa unavailable. Pin MUJOCO_GL=disable: `import mujoco` still works and no GL
    # context is created, so nothing can segfault; a render attempt raises a CATCHABLE
    # error (handled in pogo_env) and the visual cells skip gracefully.
    os.environ["MUJOCO_GL"] = "disable"
    print("Render backend: OSMesa unavailable - rendering disabled; training and "
          "plots still run, and the Meet-Pogo / video cells skip gracefully.")
    return ""


# One-call setup
def setup(drive: bool = True, install_deps: bool = True, root: str = DEFAULT_ROOT):
    """Mount Drive, install deps, set a CUDA-safe render backend, make Drive folders.

    Rendering uses OSMesa on Colab (software, CUDA-safe) so the env-preview video works
    without the egl+CUDA segfault - training is unaffected. The heavy flight-demo videos
    stay gated by cfg.RENDER (default off: watch the trained flip on your Mac via
    run_live_mac.py); the short env preview always renders.
    """
    if drive:
        mount_drive()
    if install_deps:
        install(root)

    ensure_render_backend(root)                        # CUDA-safe backend (osmesa on Colab)

    from src.config import config as cfg
    for d in ("checkpoints", "videos", "plots", "logs"):
        os.makedirs(f"{cfg.DRIVE_ROOT}/{d}", exist_ok=True)

    print(f"DRIVE_ROOT = {cfg.DRIVE_ROOT}")
    print("Setup complete - no config edit, no kernel restart.")
    return cfg.DRIVE_ROOT
