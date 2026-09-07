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
# CRITICAL: the training KERNEL is pinned to MUJOCO_GL=disable. OSMesa (software GL)
# pulls in libLLVM/libstdc++; loaded into a kernel that also imports torch +
# stable-baselines3, those native libs clash and SEGFAULT the kernel (observed exactly
# at `from stable_baselines3 import SAC` right after `import mujoco` with osmesa). With
# `disable`, `import mujoco` never loads any GL lib, so torch/SB3/training are safe.
# The env-preview video is rendered in an ISOLATED subprocess that sets MUJOCO_GL=osmesa
# and never imports torch/SB3 (see viz.plots.record_env_video_subprocess) - so libOSMesa
# never enters the training kernel. We only apt-install libOSMesa here so that child can
# render. `find_library` keeps the check instant (no torch/CUDA probe).
def _has_lib(name: str) -> bool:
    import ctypes.util
    try:
        return ctypes.util.find_library(name) is not None
    except Exception:
        return False


def ensure_render_backend(root: str = DEFAULT_ROOT) -> str:
    """Pin the KERNEL to MUJOCO_GL=disable (SB3/torch-safe) and make sure libOSMesa is
    available for the isolated render subprocess. Fast: no torch/CUDA/render probe.
    Honors an explicit MUJOCO_GL if the user set one.
    """
    if os.environ.get("MUJOCO_GL"):
        return os.environ["MUJOCO_GL"]

    if os.path.isdir("/content") and not _has_lib("OSMesa"):   # Colab: install for the subprocess
        try:
            subprocess.run(["apt-get", "install", "-y", "-qq", "libosmesa6"],
                           check=False, capture_output=True, timeout=300)
        except Exception:
            pass

    # Kernel never loads a GL lib -> no libOSMesa/LLVM vs torch/SB3 segfault. The env
    # preview renders out-of-process (osmesa on Colab, glfw on mac); other visual cells
    # skip gracefully if that subprocess can't render.
    os.environ["MUJOCO_GL"] = "disable"
    print("Render backend (kernel): disable - SB3/torch-safe; env preview renders in an "
          "isolated subprocess.")
    return "disable"


# One-call setup
def setup(drive: bool = True, install_deps: bool = True, root: str = DEFAULT_ROOT):
    """Mount Drive, install deps, pin a safe render backend, make Drive folders.

    The kernel uses MUJOCO_GL=disable so torch + stable-baselines3 import safely (loading
    libOSMesa into the kernel segfaults them). The env-preview video renders in an isolated
    subprocess instead; libOSMesa is apt-installed for that child. The heavy flight-demo
    videos stay gated by cfg.RENDER (default off: watch the trained flip on your Mac).
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
