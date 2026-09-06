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

    # Nothing probed clean. Pin MUJOCO_GL=disable: `import mujoco` still works and no
    # GL context is created, so nothing can segfault; a render attempt raises a
    # CATCHABLE error (handled in pogo_env) and the visual cells skip gracefully.
    os.environ["MUJOCO_GL"] = "disable"
    print("Render backend: none probed clean - rendering disabled; training and "
          "plots still run, and the Meet-Pogo / video cells skip gracefully.")
    return ""


# One-call setup
def setup(drive: bool = True, install_deps: bool = True, root: str = DEFAULT_ROOT):
    """Mount Drive, install deps, set the render backend, make Drive folders.

    Default workflow: TRAIN on Colab, WATCH the flip on your Mac (run_live_mac.py).
    So rendering is OFF here (cfg.RENDER) and MUJOCO_GL is pinned to "disable" - no
    GL context is ever created, so egl can't segfault next to CUDA. Set
    MACURA_RENDER=1 to render on Colab; setup() then probes for a CUDA-safe backend.
    """
    if drive:
        mount_drive()
    if install_deps:
        install(root)

    from src.config import config as cfg
    if cfg.RENDER:
        ensure_render_backend(root)                    # opt-in: pick a CUDA-safe backend
    else:
        # MUJOCO_GL=disable: `import mujoco` still works, but no GL context is ever
        # created - so no egl+CUDA segfault. Training and the matplotlib result plots
        # are unaffected; the visual cells skip cleanly.
        os.environ["MUJOCO_GL"] = "disable"
        print("Rendering: OFF (train on Colab, watch on your Mac via run_live_mac.py). "
              "Set MACURA_RENDER=1 to render on Colab.")

    for d in ("checkpoints", "videos", "plots", "logs"):
        os.makedirs(f"{cfg.DRIVE_ROOT}/{d}", exist_ok=True)

    print(f"DRIVE_ROOT = {cfg.DRIVE_ROOT}")
    print("Setup complete - no config edit, no kernel restart.")
    return cfg.DRIVE_ROOT
