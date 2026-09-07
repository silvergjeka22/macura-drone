"""Plotting — the backflip comparison figures + policy video helpers.

Pure-function library: every figure function takes data/handles, optionally saves,
and returns a matplotlib Figure. The notebook decides what to show.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt


# consistent per-algorithm colors across every figure
ALGO_COLORS = {
    "macura": "#1f77b4",   # blue  — the proposed method
    "mbpo": "#d62728",     # red   — fixed schedule (over-imagines the flight)
    "m2ac": "#2ca02c",     # green — masking
    "sac": "#7f7f7f",      # gray  — model-free reference
}


def _color(algo):
    return ALGO_COLORS.get(algo.lower(), None)


# ── a quick look at the body ──────────────────────────────────────────────────
def record_env_video(env, save_path, seconds=8, fps=30, policy="random", seed=0,
                     action_scale=0.6):
    """Render the env to an mp4 under a simple UNTRAINED policy - a quick look at Pogo
    before any learning. `policy` in {'random','still', callable(obs)->action}. Uses a
    close tracking free-camera (follows the torso in x and up), so the body fills the
    frame instead of being a speck. Resets on fall/timeout to fill the clip. Returns the
    path, or None if the env has no working renderer (headless with GL disabled)."""
    if not getattr(env, "render_enabled", True) or getattr(env, "_renderer", None) is None:
        print("record_env_video: rendering unavailable, skipping.")
        return None
    import imageio
    import mujoco
    m, d, r = env.model, env.data, env._renderer
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation, cam.azimuth = 3.6, -6.0, 90.0
    rng = np.random.default_rng(seed)
    n_act = env.action_space.shape[0]
    obs, _ = env.reset(seed=seed)
    frames = []
    for _ in range(int(seconds * fps)):
        if policy == "random":
            act = np.clip(rng.normal(0, action_scale, size=n_act), -1, 1).astype(np.float32)
        elif policy == "still":
            act = np.zeros(n_act, np.float32)
        else:
            act = policy(obs)
        obs, _, terminated, truncated, _ = env.step(act)
        cam.lookat[:] = [float(d.qpos[0]), 0.0, max(0.7, float(d.qpos[1]))]
        r.update_scene(d, camera=cam)
        frames.append(r.render())
        if terminated or truncated:
            obs, _ = env.reset()
    h, w = frames[0].shape[0] // 2 * 2, frames[0].shape[1] // 2 * 2   # even dims for h264
    frames = [f[:h, :w] for f in frames]
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    imageio.mimsave(save_path, frames, fps=fps)
    return save_path


def render_filmstrip(env, policy="random", n_frames=6, steps_between=6, seed=0,
                     title="rollout", save_path=None):
    """Capture rendered frames across a rollout and show them as a row of photos.
    `policy` in {'random', 'still', callable(obs)->action}. `env` must render().
    Returns None (skips) if the env has no working renderer, so a headless Colab
    still runs the rest of the notebook."""
    if not getattr(env, "render_enabled", True):
        print("render_filmstrip: rendering unavailable, skipping.")
        return None
    obs, _ = env.reset(seed=seed)
    frames = [env.render()]
    for _ in range(n_frames - 1):
        for _ in range(steps_between):
            act = env.action_space.sample() if policy == "random" else (
                np.zeros(env.action_space.shape[0], np.float32) if policy == "still" else policy(obs))
            obs, _, terminated, truncated, _ = env.step(act)
            if terminated or truncated:
                break
        frames.append(env.render())
    fig, axes = plt.subplots(1, len(frames), figsize=(2.6 * len(frames), 2.8))
    axes = np.atleast_1d(axes)
    for i, (ax, fr) in enumerate(zip(axes, frames)):
        ax.imshow(fr); ax.axis("off"); ax.set_title(f"step {i * steps_between}")
    fig.suptitle(title); fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


# ── comparison figures (runs is a list of run_dicts from train_one) ───────────
def _group_by_algo(runs):
    by = {}
    for r in runs:
        by.setdefault(r["algo"], []).append(r)
    return by


def _mean_std_curve(run_list, ykey):
    steps = np.asarray(run_list[0]["steps"])
    ys = np.array([r[ykey][: len(steps)] for r in run_list])
    return steps, ys.mean(axis=0), ys.std(axis=0)


def _curve_figure(runs, ykey, ylabel, title, save_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for algo, run_list in _group_by_algo(runs).items():
        steps, mean, std = _mean_std_curve(run_list, ykey)
        ax.plot(steps, mean, label=algo.upper(), color=_color(algo))
        ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color=_color(algo))
    ax.set_xlabel("real environment steps"); ax.set_ylabel(ylabel)
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    _maybe_save(fig, save_path)
    return fig


def plot_sample_efficiency(runs, save_path=None):
    """Return vs real env steps, mean +/- std over seeds, all four algorithms."""
    return _curve_figure(runs, "eval_return", "evaluation return", "Sample efficiency", save_path)


def plot_success_rate(runs, save_path=None):
    """Fraction of eval episodes that land a FULL backflip, over training (headline)."""
    return _curve_figure(runs, "eval_success_rate", "stuck-backflip rate",
                         "Backflips landed while learning", save_path)


def plot_failure_rate(runs, save_path=None):
    """Fraction of eval episodes that collapse / face-plant, over training."""
    return _curve_figure(runs, "eval_failure_rate", "faceplant rate",
                         "Faceplant rate while learning", save_path)


def plot_final_quality(runs, save_path=None):
    """Bar chart of best evaluation return per algorithm (mean +/- std over seeds)."""
    by = _group_by_algo(runs)
    algos, means, stds = [], [], []
    for algo, run_list in by.items():
        bests = [max(r["eval_return"]) for r in run_list]
        algos.append(algo.upper()); means.append(np.mean(bests)); stds.append(np.std(bests))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(algos, means, yerr=stds, capsize=5, color=[ALGO_COLORS.get(a.lower()) for a in algos])
    ax.set_ylabel("best evaluation return"); ax.set_title("Final policy quality")
    ax.grid(alpha=0.3, axis="y")
    _maybe_save(fig, save_path)
    return fig


def plot_rollout_depth(macura_run, save_path=None):
    """MACURA signature: mean rollout length (grows) and kappa (falls) over training."""
    fig, ax1 = plt.subplots(figsize=(7, 5))
    if macura_run["rollout_length"]:
        rs, rl = zip(*macura_run["rollout_length"])
        ax1.plot(rs, rl, "C0", label="mean rollout length")
        ax1.set_ylabel("mean rollout length", color="C0")
    ax1.set_xlabel("real environment steps")
    if macura_run["kappa"]:
        ks, kv = zip(*macura_run["kappa"])
        ax2 = ax1.twinx()
        ax2.plot(ks, kv, "C3", label="kappa")
        ax2.set_ylabel("kappa", color="C3")
    ax1.set_title("MACURA: adaptive rollout length & kappa"); ax1.grid(alpha=0.3)
    _maybe_save(fig, save_path)
    return fig


# ── policy video ──────────────────────────────────────────────────────────────
def record_policy_video(agent, env, seconds=8, fps=30, save_path=None):
    """Record a deterministic evaluation clip (~`seconds`) of a trained SB3 agent.
    Resets on crash/timeout so the clip fills the duration. `env` must render()."""
    import imageio
    pol = lambda o: agent.predict(np.asarray(o, np.float32), deterministic=True)[0]
    obs, _ = env.reset()
    frames = [_pad16(env.render())]
    for _ in range(int(seconds * fps)):
        obs, _, terminated, truncated, _ = env.step(pol(obs))
        frames.append(_pad16(env.render()))
        if terminated or truncated:
            obs, _ = env.reset()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        imageio.mimsave(save_path, frames, fps=fps)
    return save_path


def _pad16(frame):
    h, w = frame.shape[0] // 16 * 16, frame.shape[1] // 16 * 16
    return frame[:h, :w]


def stitch_videos_grid(video_paths: dict, save_path: str, fps=30, cols=2, downscale=2):
    """Side-by-side reel: tile the given {label: mp4_path} clips into one grid video
    with each label burned in. Shorter clips hold their last frame."""
    import imageio
    clips, names = [], []
    for name, path in video_paths.items():
        rd = imageio.get_reader(path)
        frames = [np.asarray(f)[::downscale, ::downscale, :3] for f in rd]
        rd.close()
        if frames:
            clips.append(frames); names.append(name)
    if not clips:
        raise ValueError("no readable clips in video_paths")

    h = max(c[0].shape[0] for c in clips)
    w = max(c[0].shape[1] for c in clips)
    T = max(len(c) for c in clips)
    rows = int(np.ceil(len(clips) / cols))

    try:
        from PIL import Image, ImageDraw

        def _label(img, text):
            im = Image.fromarray(img); d = ImageDraw.Draw(im)
            d.rectangle([0, 0, 14 + 8 * len(text), 22], fill=(0, 0, 0))
            d.text((7, 5), text, fill=(255, 255, 255))
            return np.asarray(im)
    except ImportError:
        def _label(img, text):
            return img

    def _tile(clip, t, name):
        f = clip[min(t, len(clip) - 1)]
        out = np.zeros((h, w, 3), np.uint8)
        out[: f.shape[0], : f.shape[1]] = f
        return _label(out, name.upper())

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    writer = imageio.get_writer(save_path, fps=fps)
    blank = np.zeros((h, w, 3), np.uint8)
    for t in range(T):
        tiles = [_tile(c, t, n) for c, n in zip(clips, names)]
        tiles += [blank] * (rows * cols - len(tiles))
        grid = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])
        writer.append_data(_pad16(grid))
    writer.close()
    return save_path


def _maybe_save(fig, save_path):
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
