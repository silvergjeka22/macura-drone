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


# One-call setup
def setup(drive: bool = True, install_deps: bool = True, root: str = DEFAULT_ROOT):
    """Mount Drive, install deps, and make the Drive result folders."""
    os.environ.setdefault("MUJOCO_GL", "egl")          # headless rendering on Colab
    if drive:
        mount_drive()
    if install_deps:
        install(root)

    from src.config import config as cfg
    for d in ("checkpoints", "videos", "plots", "logs"):
        os.makedirs(f"{cfg.DRIVE_ROOT}/{d}", exist_ok=True)

    print(f"DRIVE_ROOT = {cfg.DRIVE_ROOT}")
    print("Setup complete - no config edit, no kernel restart.")
    return cfg.DRIVE_ROOT
