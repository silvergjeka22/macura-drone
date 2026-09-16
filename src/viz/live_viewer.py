"""Real-time macOS viewer: watch a trained drone policy fly live.

Train on Kaggle (GPU) -> copy the best `.zip` to your Mac -> watch it here in a native
MuJoCo window, in real time. Inference is cheap, so it runs smoothly on CPU (the drone
body ships in the repo as envs/assets/drone.xml).

macOS note: MuJoCo's interactive viewer must run on the main thread, so launch the entry
script with `mjpython` (bundled with the `mujoco` pip package), NOT plain `python`:

    mjpython run_live_mac.py --ckpt runs/checkpoints/macura_seed0_best.zip

Pure-function library; the entry point `run_live_mac.py` calls these.

Public functions:
    load_policy(ckpt_path, device)                 -> SB3 SAC agent
    fly_policy(ckpt_path, env_cfg, ...)            open a live window, fly one policy
    fly_sequence(ckpts, env_cfg, ...)             fly several policies one after another
"""

from __future__ import annotations

import time
import numpy as np

from src.envs import drone_env


def load_policy(ckpt_path: str, device: str = "cpu"):
    """Load a trained SB3 SAC policy from a .zip (device-agnostic; CPU is fine)."""
    from stable_baselines3 import SAC
    return SAC.load(ckpt_path, device=device)


def fly_policy(ckpt_path: str, env_cfg: dict, seconds: float = 20.0, seed: int = 0,
               realtime: bool = True, device: str = "cpu"):
    """Open a live MuJoCo window and fly the trained drone policy.

    Resets on a crash/timeout so the window keeps showing attempts for `seconds`.
    `env_cfg` is the `env:` sub-config (the same one training used). Returns the best
    (smallest) final distance-to-target seen.
    """
    import mujoco
    import mujoco.viewer

    agent = load_policy(ckpt_path, device)
    env = drone_env.DroneTargetEnv(env_cfg, seed=seed, render=False)
    dt = float(env.model.opt.timestep) * env.action_repeat     # wall-clock per agent step
    obs, _ = env.reset(seed=seed)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        t_end = time.time() + seconds
        best_dist = np.inf
        while viewer.is_running() and time.time() < t_end:
            tick = time.time()
            act = agent.predict(np.asarray(obs, np.float32), deterministic=True)[0]
            obs, _, terminated, truncated, info = env.step(act)
            best_dist = min(best_dist, info.get("dist", np.inf))
            viewer.sync()
            if terminated or truncated:
                obs, _ = env.reset()
            if realtime:
                slack = dt - (time.time() - tick)
                if slack > 0:
                    time.sleep(slack)
    env.close()
    return best_dist


def fly_sequence(ckpts: dict, env_cfg: dict, seconds_each: float = 12.0, seed: int = 0,
                 device: str = "cpu"):
    """Fly several policies one after another in their own windows, so you can watch
    MACURA vs MBPO vs M2AC vs SAC in turn. `ckpts` maps label -> .zip path."""
    for label, path in ckpts.items():
        print(f"--- {label}: {path} (close the window to continue) ---")
        try:
            dist = fly_policy(path, env_cfg, seconds=seconds_each, seed=seed, device=device)
            print(f"    {label}: best distance-to-target {dist:.2f} m")
        except FileNotFoundError:
            print(f"    {label}: checkpoint not found, skipping")
