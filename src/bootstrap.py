"""Kaggle kernel setup: install deps, pick a safe render backend, make output folders.

Called once at the top of the notebook (## Setup). Everything runs on a Kaggle GPU kernel;
/kaggle/working is the only writable folder Kaggle saves as the kernel's downloadable output.
"""

import os
import subprocess
import sys

# Where the project source (src/) is cloned on the kernel (see the notebook's "Get the code").
DEFAULT_ROOT = "/kaggle/working/macura-backflip"


def install(root: str = DEFAULT_ROOT):
    """Install the kernel's Python deps from requirements.txt (mujoco, gymnasium,
    stable-baselines3, ...). Kaggle preinstalls numpy/torch, so this only adds what is missing."""
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                    os.path.join(root, "requirements.txt")], check=True)
    print("Requirements installed.")


def _has_lib(name: str) -> bool:
    import ctypes.util
    try:
        return ctypes.util.find_library(name) is not None
    except Exception:
        return False


# Headless rendering (a Kaggle kernel has no display).
# CRITICAL: the training kernel is pinned to MUJOCO_GL=disable. OSMesa (software GL) pulls in
# libLLVM/libstdc++; loaded into a kernel that also imports torch + stable-baselines3, those
# native libs clash and SEGFAULT the kernel (right at `from stable_baselines3 import SAC` after
# `import mujoco` with osmesa). With `disable`, `import mujoco` never loads a GL lib, so
# torch/SB3/training are safe. The optional env-preview video renders in an ISOLATED subprocess
# that sets MUJOCO_GL=osmesa and never imports torch/SB3, so libOSMesa never enters the training
# kernel; we best-effort apt-install libOSMesa here only so that child process can render.
def ensure_render_backend() -> str:
    """Pin the kernel to MUJOCO_GL=disable (torch/SB3-safe) and best-effort make libOSMesa
    available for the isolated preview subprocess. Honors an explicit MUJOCO_GL if you set one."""
    if os.environ.get("MUJOCO_GL"):
        return os.environ["MUJOCO_GL"]

    if not _has_lib("OSMesa"):                          # only for the isolated preview subprocess
        try:
            subprocess.run(["apt-get", "install", "-y", "-qq", "libosmesa6"],
                           check=False, capture_output=True, timeout=300)
        except Exception:
            pass

    os.environ["MUJOCO_GL"] = "disable"
    print("Render backend (kernel): disable - torch/SB3-safe; the preview renders out-of-process.")
    return "disable"


def setup(install_deps: bool = True, root: str = DEFAULT_ROOT):
    """One-call setup for the Kaggle kernel: install deps, pin a safe render backend, and make
    the output folders (checkpoints/videos/plots/logs) under OUTPUT_ROOT (/kaggle/working/runs).
    The heavy flight-demo videos stay gated by cfg.RENDER (default off: watch on your Mac)."""
    if install_deps:
        install(root)

    ensure_render_backend()

    from src.config import config as cfg
    for d in ("checkpoints", "videos", "plots", "logs"):
        os.makedirs(f"{cfg.OUTPUT_ROOT}/{d}", exist_ok=True)

    print(f"OUTPUT_ROOT = {cfg.OUTPUT_ROOT}")
    print("Setup complete.")
    return cfg.OUTPUT_ROOT
