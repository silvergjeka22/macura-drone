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
# MuJoCo picks its GL backend from MUJOCO_GL the first time a rendering context is
# created, and a bad choice does not raise - it SEGFAULTS and kills the kernel
# (this is the classic "Meet Pogo" cell crash). So we probe each backend in an
# ISOLATED SUBPROCESS: a segfault there only kills the probe, never the notebook,
# and we keep the first backend that actually renders a frame.
def _probe_gl(backend: str, root: str) -> bool:
    asset = os.path.join(root, "src", "envs", "assets", "pogo.xml")
    code = (
        "import os; os.environ['MUJOCO_GL'] = %r\n"
        "import mujoco\n"
        "m = mujoco.MjModel.from_xml_path(%r)\n"
        "d = mujoco.MjData(m)\n"
        "r = mujoco.Renderer(m, 64, 64)\n"
        "mujoco.mj_forward(m, d)\n"
        "r.update_scene(d); r.render(); r.close()\n"
        "print('OK')\n" % (backend, asset)
    )
    try:
        out = subprocess.run([sys.executable, "-c", code],
                             capture_output=True, text=True, timeout=180)
        return out.returncode == 0 and "OK" in out.stdout
    except Exception:
        return False


def ensure_render_backend(root: str = DEFAULT_ROOT) -> str:
    """Pick a MuJoCo GL backend that actually works and set MUJOCO_GL to it.

    Honors an explicit MUJOCO_GL if the user set one. Otherwise tries egl (GPU,
    fast), then osmesa (software, rock-solid) - apt-installing the OSMesa library
    on Colab if egl failed - then glfw. Returns the chosen backend, or "" if none
    rendered (training still works; only the visualization cells are unavailable).
    """
    if os.environ.get("MUJOCO_GL"):
        return os.environ["MUJOCO_GL"]

    if _probe_gl("egl", root):
        os.environ["MUJOCO_GL"] = "egl"
        print("Render backend: egl (GPU).")
        return "egl"

    # egl unavailable/unstable -> make sure the OSMesa software renderer is present
    if os.path.isdir("/content"):                       # Colab: apt is available as root
        try:
            subprocess.run(["apt-get", "install", "-y", "-qq", "libosmesa6"],
                           check=False, capture_output=True, timeout=300)
        except Exception:
            pass
    for backend in ("osmesa", "glfw"):
        if _probe_gl(backend, root):
            os.environ["MUJOCO_GL"] = backend
            print(f"Render backend: {backend}"
                  f"{' (software)' if backend == 'osmesa' else ''}.")
            return backend

    print("Render backend: none available - training/plots still run; "
          "the Meet-Pogo and video cells will be skipped.")
    return ""


# One-call setup
def setup(drive: bool = True, install_deps: bool = True, root: str = DEFAULT_ROOT):
    """Mount Drive, install deps, pick a working render backend, make Drive folders."""
    if drive:
        mount_drive()
    if install_deps:
        install(root)

    ensure_render_backend(root)                        # sets MUJOCO_GL (segfault-safe)

    from src.config import config as cfg
    for d in ("checkpoints", "videos", "plots", "logs"):
        os.makedirs(f"{cfg.DRIVE_ROOT}/{d}", exist_ok=True)

    print(f"DRIVE_ROOT = {cfg.DRIVE_ROOT}")
    print("Setup complete - no config edit, no kernel restart.")
    return cfg.DRIVE_ROOT
