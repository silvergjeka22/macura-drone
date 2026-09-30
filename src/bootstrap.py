"""Kaggle setup: install the requirements, choose a safe render backend, make the output folders."""

import os
import subprocess
import sys

DEFAULT_ROOT = "/kaggle/working/macura-drone"


def install(root: str = DEFAULT_ROOT):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                    os.path.join(root, "requirements.txt")], check=True)
    print("Requirements installed.")


def _has_lib(name: str) -> bool:
    import ctypes.util
    try:
        return ctypes.util.find_library(name) is not None
    except Exception:
        return False


def ensure_render_backend() -> str:
    if os.environ.get("MUJOCO_GL"):
        return os.environ["MUJOCO_GL"]
    if not _has_lib("OSMesa"):
        try:
            subprocess.run(["apt-get", "install", "-y", "-qq", "libosmesa6"],
                           check=False, capture_output=True, timeout=300)
        except Exception:
            pass
    os.environ["MUJOCO_GL"] = "disable"
    return "disable"


def setup(install_deps: bool = True, root: str = DEFAULT_ROOT):
    if install_deps:
        install(root)
    ensure_render_backend()
    from src.config import config as cfg
    for d in ("checkpoints", "videos", "plots", "logs"):
        os.makedirs(f"{cfg.OUTPUT_ROOT}/{d}", exist_ok=True)
    print(f"task {cfg.TASK} | {cfg.TOTAL_ENV_STEPS} steps | seeds {cfg.SEEDS} | "
          f"algorithms {cfg.ALGORITHMS} | output {cfg.OUTPUT_ROOT}")
    return cfg.OUTPUT_ROOT
