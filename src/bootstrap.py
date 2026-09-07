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
    # Reproduce the REAL kernel condition: torch has already created a CUDA context
    # by the time the render cell runs, and an EGL rendering context on a GPU that
    # already has a live CUDA context is exactly what segfaults. So the probe inits
    # CUDA first - otherwise egl passes here but dies in the notebook (false positive).
    code = (
        "import os; os.environ['MUJOCO_GL'] = %r\n"
        "try:\n"
        "    import torch\n"
        "    if torch.cuda.is_available():\n"
        "        torch.zeros(1, device='cuda'); torch.cuda.synchronize()\n"
        "except Exception:\n"
        "    pass\n"
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
    """Set MUJOCO_GL to a CUDA-SAFE render backend so headless video works AND training
    does not crash.

    On Colab we use OSMesa (software rendering): it renders headless and, being CPU-only,
    does NOT fight the live CUDA context the way egl does (egl next to CUDA segfaults the
    kernel). We apt-install libOSMesa first, then verify it in an isolated subprocess.
    Locally (mac) we use glfw. If nothing works we pin `disable` - `import mujoco` still
    succeeds, no GL context is created, and the visual cells skip gracefully.
    Honors an explicit MUJOCO_GL if the user set one.
    """
    if os.environ.get("MUJOCO_GL"):
        return os.environ["MUJOCO_GL"]

    if os.path.isdir("/content"):                       # Colab: apt is available as root
        try:
            subprocess.run(["apt-get", "install", "-y", "-qq", "libosmesa6"],
                           check=False, capture_output=True, timeout=300)
        except Exception:
            pass
        if _probe_gl("osmesa", root):
            os.environ["MUJOCO_GL"] = "osmesa"
            print("Render backend: osmesa (software, CUDA-safe).")
            return "osmesa"
    elif _probe_gl("glfw", root):                       # local (mac/linux with a display)
        os.environ["MUJOCO_GL"] = "glfw"
        print("Render backend: glfw.")
        return "glfw"

    # Nothing probed clean. Pin MUJOCO_GL=disable: `import mujoco` still works and no
    # GL context is created, so nothing can segfault; a render attempt raises a
    # CATCHABLE error (handled in pogo_env) and the visual cells skip gracefully.
    os.environ["MUJOCO_GL"] = "disable"
    print("Render backend: none probed clean - rendering disabled; training and "
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
