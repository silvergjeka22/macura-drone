"""Plotting — the algorithm-comparison figures + policy/env video helpers.

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
    """Render the env to an mp4 under a simple UNTRAINED policy - a quick look at the drone
    before any learning. `policy` in {'random','still', callable(obs)->action}. Uses a close
    free-camera that follows the body's world position, so it fills the frame. Resets on
    crash/timeout to fill the clip. Returns the path, or None if the env has no working
    renderer (headless with GL disabled)."""
    if not getattr(env, "render_enabled", True) or getattr(env, "_renderer", None) is None:
        print("record_env_video: rendering unavailable, skipping.")
        return None
    import imageio
    import mujoco
    m, d, r = env.model, env.data, env._renderer
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation, cam.azimuth = 4.0, -15.0, 90.0
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
        cam.lookat[:] = d.xpos[1]                 # follow the main body's world position (any env)
        r.update_scene(d, camera=cam)
        frames.append(r.render())
        if terminated or truncated:
            obs, _ = env.reset()
    h, w = frames[0].shape[0] // 2 * 2, frames[0].shape[1] // 2 * 2   # even dims for h264
    frames = [f[:h, :w] for f in frames]
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    imageio.mimsave(save_path, frames, fps=fps)
    return save_path


def record_env_video_subprocess(root, save_path, seconds=6, policy="random", seed=0,
                                backend="osmesa"):
    """Render the env preview in a SEPARATE process with a software GL backend, then
    return `save_path` (or None on failure).

    Why a subprocess: OSMesa (software GL) loads libLLVM/libstdc++; in a kernel that also
    imports torch + stable-baselines3 those clash and SEGFAULT. The training kernel is
    therefore pinned to MUJOCO_GL=disable, and this child renders with MUJOCO_GL=osmesa
    while importing ONLY mujoco/gymnasium/numpy/imageio (never torch or SB3) - so libOSMesa
    never touches the training kernel. `backend` is 'osmesa' on Kaggle, 'glfw' locally.
    """
    import subprocess
    import sys
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    child = (
        "import os; os.environ['MUJOCO_GL'] = %r\n"
        "import sys; sys.path.insert(0, %r)\n"
        "from src.config import config as cfg\n"
        "from src.envs import drone_env\n"
        "from src.viz import plots\n"
        "env, _, _ = drone_env.make_env(cfg.ENV, seed=%d, render=True)\n"
        "p = plots.record_env_video(env, %r, seconds=%d, policy=%r, seed=%d)\n"
        "env.close()\n"
        "print('OK' if p else 'FAIL')\n"
        % (backend, root, seed, save_path, int(seconds), policy, seed)
    )
    try:
        r = subprocess.run([sys.executable, "-c", child],
                           capture_output=True, text=True, timeout=600)
        if r.returncode == 0 and os.path.exists(save_path):
            return save_path
        print(f"env-video subprocess failed (rc={r.returncode}):", (r.stderr or "")[-600:])
        return None
    except Exception as e:
        print("env-video subprocess error:", e)
        return None


# ── comparison figures (runs is a list of run_dicts from train_one) ───────────
def _group_by_algo(runs):
    by = {}
    for r in runs:
        by.setdefault(r["algo"], []).append(r)
    return by


def _mean_std_curve(run_list, ykey):
    # robust to seeds with slightly different numbers of eval points: clip every run
    # (and the step axis) to the shortest, so mean/std never hit a ragged-array error.
    n = min(min(len(r["steps"]), len(r[ykey])) for r in run_list)
    steps = np.asarray(run_list[0]["steps"][:n])
    ys = np.array([r[ykey][:n] for r in run_list])
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
    """Return vs real env steps, mean +/- std over seeds, all four algorithms. A dashed
    horizontal line marks model-free SAC's FINAL return (the paper's SAC reference): the
    model-based methods aim to reach it in far fewer steps."""
    fig = _curve_figure(runs, "eval_return", "evaluation return",
                        "Sample efficiency (return vs real steps)", None)
    ax = fig.axes[0]
    by = _group_by_algo(runs)
    if "sac" in by:
        sac_final = np.mean([r["eval_return"][-1] for r in by["sac"] if r["eval_return"]])
        ax.axhline(sac_final, ls="--", color=_color("sac"), lw=1.5,
                   label="SAC (final)")
        ax.legend()
    _maybe_save(fig, save_path)
    return fig


def plot_success_rate(runs, save_path=None):
    """Fraction of eval episodes that SUCCEED (drone: reached & held the target), over training."""
    return _curve_figure(runs, "eval_success_rate", "success rate",
                         "Task success while learning", save_path)


def plot_failure_rate(runs, save_path=None):
    """Fraction of eval episodes that CRASH (drone: flipped / hit ground / flew away), over training."""
    return _curve_figure(runs, "eval_failure_rate", "crash rate",
                         "Crash rate while learning", save_path)


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


# ── (step, value) diagnostic series logged during training ────────────────────
def _series_mean_std(run_list, key):
    """Mean +/- std over seeds of a logged [(step, value), ...] diagnostic series.
    Aligns on the shared step axis (clipped to the shortest run)."""
    series = [np.asarray(r[key], dtype=float) for r in run_list if r.get(key)]
    if not series:
        return None, None, None
    n = min(len(s) for s in series)
    steps = series[0][:n, 0]
    vals = np.stack([s[:n, 1] for s in series])
    return steps, vals.mean(axis=0), vals.std(axis=0)


def plot_imagined_horizon(runs, save_path=None):
    """How far each model-based method IMAGINES over training: MACURA adapts its rollout
    length to model uncertainty (grows as the model earns trust), MBPO follows a fixed
    ramp, M2AC holds a fixed horizon. This is the core mechanism the paper is about."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for algo, run_list in _group_by_algo(runs).items():
        if algo == "sac":
            continue
        steps, mean, std = _series_mean_std(run_list, "rollout_length")
        if steps is None:
            continue
        ax.plot(steps, mean, label=algo.upper(), color=_color(algo))
        ax.fill_between(steps, mean - std, mean + std, alpha=0.15, color=_color(algo))
    ax.set_xlabel("real environment steps"); ax.set_ylabel("imagined rollout length (steps)")
    ax.set_title("How far each method trusts the model"); ax.legend(); ax.grid(alpha=0.3)
    _maybe_save(fig, save_path)
    return fig


def plot_uncertainty(macura_runs, save_path=None):
    """MACURA's model uncertainty story: the first-step GJS uncertainty of fresh states
    (how wrong the model could be) against the adaptive trust threshold kappa, over
    training. `macura_runs` is one macura run_dict or a list of them (averaged over seeds)."""
    run_list = macura_runs if isinstance(macura_runs, list) else [macura_runs]
    fig, ax = plt.subplots(figsize=(7, 5))
    for key, color, label in (("base_uncertainty", "#9467bd", "first-step GJS uncertainty"),
                              ("kappa", "#d62728", "kappa (trust threshold)")):
        steps, mean, std = _series_mean_std(run_list, key)
        if steps is None:
            continue
        ax.plot(steps, mean, color=color, label=label)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.15, color=color)
    ax.set_xlabel("real environment steps"); ax.set_ylabel("GJS uncertainty")
    ax.set_title("MACURA: model uncertainty vs the adaptive trust threshold")
    ax.legend(); ax.grid(alpha=0.3)
    _maybe_save(fig, save_path)
    return fig


def plot_rollout_distribution(macura_run, save_path=None):
    """Histogram of MACURA's per-rollout truncation lengths in the latest rollout round:
    a fixed-horizon method would be a single spike at t_max; MACURA spreads across lengths
    because it stops each branch where THAT branch leaves the trust region."""
    hist = macura_run.get("rollout_len_hist") or []
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if hist:
        hi = int(max(hist))
        ax.hist(hist, bins=np.arange(-0.5, hi + 1.5, 1.0),
                color=_color("macura"), edgecolor="white")
        ax.axvline(float(np.mean(hist)), ls="--", color="k", lw=1.2,
                   label=f"mean = {np.mean(hist):.1f}")
        ax.legend()
    ax.set_xlabel("rollout length before truncation (steps)")
    ax.set_ylabel("number of branched rollouts")
    ax.set_title("MACURA: per-rollout truncation lengths")
    ax.grid(alpha=0.3, axis="y")
    _maybe_save(fig, save_path)
    return fig


def plot_summary_table(runs, save_path=None):
    """A compact results table (one row per algorithm, mean +/- std over seeds):
    best return, final return, final success (reached-and-held) rate, final crash rate."""
    order = ["macura", "mbpo", "m2ac", "sac"]
    by = _group_by_algo(runs)
    col_labels = ["algorithm", "best return", "final return", "success", "crash"]
    rows, colors = [], []

    def ms(vals):
        vals = [v for v in vals if v is not None]
        return f"{np.mean(vals):.1f} +/- {np.std(vals):.1f}" if vals else "-"

    for algo in [a for a in order if a in by] + [a for a in by if a not in order]:
        rl = by[algo]
        best = [max(r["eval_return"]) for r in rl if r["eval_return"]]
        final = [r["eval_return"][-1] for r in rl if r["eval_return"]]
        succ = [r["eval_success_rate"][-1] for r in rl if r.get("eval_success_rate")]
        crash = [r["eval_failure_rate"][-1] for r in rl if r.get("eval_failure_rate")]
        rows.append([algo.upper(), ms(best), ms(final),
                     f"{np.mean(succ):.2f}" if succ else "-",
                     f"{np.mean(crash):.2f}" if crash else "-"])
        colors.append(_color(algo))

    fig, ax = plt.subplots(figsize=(8, 0.6 + 0.5 * len(rows)))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 1.5)
    for j in range(len(col_labels)):                       # header row bold
        tbl[0, j].set_text_props(weight="bold")
    for i, c in enumerate(colors, start=1):                # tint the algorithm cell
        if c:
            tbl[i, 0].set_facecolor(c); tbl[i, 0].set_text_props(color="white", weight="bold")
    ax.set_title("Results summary (mean +/- std over seeds)", pad=12)
    _maybe_save(fig, save_path)
    return fig


def plot_overview(runs, save_path=None):
    """One paper-style 2x2 panel: return, success rate, crash rate, and imagined horizon
    -- the whole comparison at a glance."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    panels = [("eval_return", "evaluation return", "Return"),
              ("eval_success_rate", "success rate", "Reached & held target"),
              ("eval_failure_rate", "crash rate", "Crash rate")]
    for ax, (ykey, ylabel, title) in zip(axes.flat[:3], panels):
        for algo, run_list in _group_by_algo(runs).items():
            steps, mean, std = _mean_std_curve(run_list, ykey)
            ax.plot(steps, mean, label=algo.upper(), color=_color(algo))
            ax.fill_between(steps, mean - std, mean + std, alpha=0.18, color=_color(algo))
        ax.set_xlabel("real environment steps"); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes.flat[3]
    for algo, run_list in _group_by_algo(runs).items():
        if algo == "sac":
            continue
        steps, mean, std = _series_mean_std(run_list, "rollout_length")
        if steps is None:
            continue
        ax.plot(steps, mean, label=algo.upper(), color=_color(algo))
        ax.fill_between(steps, mean - std, mean + std, alpha=0.15, color=_color(algo))
    ax.set_xlabel("real environment steps"); ax.set_ylabel("imagined rollout length")
    ax.set_title("How far each method trusts the model"); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    fig.suptitle("MACURA vs MBPO vs M2AC vs SAC -- drone recover-and-reach", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    _maybe_save(fig, save_path)
    return fig


# ── robust aggregation: IQM + stratified bootstrap CIs (rliable-style, pure numpy) ────
def _iqm(vals):
    """Interquartile mean: mean of the middle 50% (drop the top/bottom 25%)."""
    v = np.sort(np.asarray(vals, dtype=float))
    k = int(len(v) * 0.25)
    core = v[k:len(v) - k] if len(v) - 2 * k >= 1 else v
    return float(core.mean())


def _agg_ci(vals, kind="iqm", n_boot=2000, alpha=0.05, seed=0):
    """Center statistic (iqm or mean) of `vals` over seeds, with a bootstrap CI.
    Returns (center, lo, hi)."""
    vals = np.asarray(vals, dtype=float)
    stat = _iqm if kind == "iqm" else (lambda a: float(np.mean(a)))
    center = stat(vals)
    if len(vals) < 3:
        return center, center, center
    rng = np.random.default_rng(seed)
    boots = [stat(rng.choice(vals, size=len(vals), replace=True)) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return center, float(lo), float(hi)


def _stack(run_list, ykey):
    """(n_seeds, n_evals) matrix of a per-eval metric, clipped to the shortest run."""
    n = min(min(len(r["steps"]), len(r[ykey])) for r in run_list)
    steps = np.asarray(run_list[0]["steps"][:n])
    mat = np.array([r[ykey][:n] for r in run_list])
    return steps, mat


def _agg_curve(mat, kind):
    """Per-eval-point center + CI band across seeds (columns of `mat`)."""
    c, lo, hi = [], [], []
    for j in range(mat.shape[1]):
        cc, ll, hh = _agg_ci(mat[:, j], kind=kind, seed=j)
        c.append(cc); lo.append(ll); hi.append(hh)
    return np.array(c), np.array(lo), np.array(hi)


def plot_iqm_efficiency(runs, save_path=None):
    """Sample efficiency with the IQM (interquartile mean) over seeds + 95% bootstrap CIs -
    the rliable-standard way to compare runs at a small seed count: a real MACURA edge shows
    as a higher IQM with a tighter, separated band even when raw means overlap."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for algo, run_list in _group_by_algo(runs).items():
        steps, mat = _stack(run_list, "eval_return")
        c, lo, hi = _agg_curve(mat, "iqm")
        ax.plot(steps, c, label=algo.upper(), color=_color(algo))
        ax.fill_between(steps, lo, hi, alpha=0.18, color=_color(algo))
    ax.set_xlabel("real environment steps"); ax.set_ylabel("IQM evaluation return")
    ax.set_title("Sample efficiency - IQM +/- 95% bootstrap CI"); ax.legend(); ax.grid(alpha=0.3)
    _maybe_save(fig, save_path)
    return fig


def plot_crash_rate_ci(runs, save_path=None):
    """Crash rate over training, mean over seeds + 95% bootstrap CI - MACURA's most dependable
    edge (it avoids the model-exploitation blow-ups that make MBPO crash)."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for algo, run_list in _group_by_algo(runs).items():
        steps, mat = _stack(run_list, "eval_failure_rate")
        c, lo, hi = _agg_curve(mat, "mean")
        ax.plot(steps, c, label=algo.upper(), color=_color(algo))
        ax.fill_between(steps, lo, hi, alpha=0.18, color=_color(algo))
    ax.set_xlabel("real environment steps"); ax.set_ylabel("crash rate")
    ax.set_title("Crash rate - mean +/- 95% bootstrap CI"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_ylim(-0.02, 1.02)
    _maybe_save(fig, save_path)
    return fig


def plot_steps_to_threshold(runs, threshold=None, frac=0.6, save_path=None):
    """Data efficiency: steps to first reach a return threshold, per algorithm (median over
    seeds, IQR whiskers; lower = better). If `threshold` is None it is set to `frac` of the
    best mean return reached by any algorithm. Seeds that never reach it are marked censored."""
    by = _group_by_algo(runs)
    if threshold is None:
        best = max(max(r["eval_return"]) for rl in by.values() for r in rl)
        worst = min(min(r["eval_return"]) for rl in by.values() for r in rl)
        threshold = worst + frac * (best - worst)
    order = [a for a in ["macura", "mbpo", "m2ac", "sac"] if a in by] + \
            [a for a in by if a not in ("macura", "mbpo", "m2ac", "sac")]
    labels, meds, los, his, cens = [], [], [], [], []
    for algo in order:
        steps_to = []
        n_cens = 0
        for r in by[algo]:
            st = np.asarray(r["steps"]); ret = np.asarray(r["eval_return"])
            hit = np.where(ret >= threshold)[0]
            if len(hit):
                steps_to.append(float(st[hit[0]]))
            else:
                steps_to.append(float(st[-1])); n_cens += 1     # censored: never reached
        steps_to = np.array(steps_to)
        labels.append(algo.upper()); meds.append(np.median(steps_to))
        los.append(np.median(steps_to) - np.percentile(steps_to, 25))
        his.append(np.percentile(steps_to, 75) - np.median(steps_to))
        cens.append(n_cens)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    xs = np.arange(len(labels))
    ax.bar(xs, meds, yerr=[los, his], capsize=5,
           color=[_color(l.lower()) for l in labels])
    for i, (m, n) in enumerate(zip(meds, cens)):
        if n:
            ax.text(i, m, f"{n} censored", ha="center", va="bottom", fontsize=8, color="0.3")
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_ylabel("steps to reach threshold (median, IQR)")
    ax.set_title(f"Data efficiency - steps to return >= {threshold:.0f}  (lower = better)")
    ax.grid(alpha=0.3, axis="y")
    _maybe_save(fig, save_path)
    return fig


# ── policy video ──────────────────────────────────────────────────────────────
def record_policy_video(ckpt_path, cfg, save_path, seconds=8, seed=999, label=None, device="cpu"):
    """Save an mp4 of a trained policy flying, in a way that is safe on a headless GPU box.
    Step 1: roll the policy out here to record the body pose per frame (SB3, no graphics).
    Step 2: render those frames in a separate mujoco-only process (graphics never touch the
    training kernel). Returns save_path, or None if rendering is unavailable (run still ok)."""
    import tempfile
    try:
        qpos, mocap, fps = _policy_trajectory(ckpt_path, cfg["env"], seconds, seed, device)
    except Exception as e:
        print("policy rollout failed:", e)
        return None
    frames_file = tempfile.mktemp(suffix=".npz")
    np.savez(frames_file, qpos=qpos, mocap=mocap)
    return _render_trajectory_subprocess(frames_file, save_path, int(fps), label or "")


def _policy_trajectory(ckpt_path, env_cfg, seconds, seed, device):
    """Run the trained policy (deterministic) and return (qpos frames, mocap frames, fps).
    No graphics here, so SB3 and MuJoCo coexist safely."""
    from stable_baselines3 import SAC
    from src.envs import drone_env
    env, _, _ = drone_env.make_env(env_cfg, seed=seed, render=False)
    agent = SAC.load(ckpt_path, device=device)
    fps = int(round(1.0 / (env.model.opt.timestep * env.action_repeat)))
    obs, _ = env.reset()
    qpos, mocap = [], []
    for _ in range(int(seconds * fps)):
        act = agent.predict(np.asarray(obs, np.float32), deterministic=True)[0]
        obs, _, term, trunc, _ = env.step(act)
        qpos.append(np.array(env.data.qpos, np.float64))
        mocap.append(np.array(env.data.mocap_pos, np.float64))
        if term or trunc:
            obs, _ = env.reset()
    env.close()
    return np.array(qpos), np.array(mocap), fps


# child script: renders a saved pose trajectory. Imports ONLY mujoco (never torch/SB3), so
# software graphics can never crash the training kernel.
_RENDER_CHILD = '''
import os, sys
os.environ["MUJOCO_GL"] = "%(backend)s"
import numpy as np, mujoco, imageio
d = np.load("%(frames)s")
qpos, mocap = d["qpos"], d["mocap"]
m = mujoco.MjModel.from_xml_path("%(asset)s")
data = mujoco.MjData(m)
r = mujoco.Renderer(m, height=480, width=640)
cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.distance, cam.elevation, cam.azimuth = 4.5, -20.0, 90.0
label = "%(label)s"
try:
    from PIL import Image, ImageDraw
except Exception:
    Image = None
frames = []
for i in range(len(qpos)):
    data.qpos[:] = qpos[i]
    if mocap.shape[1] > 0:
        data.mocap_pos[:] = mocap[i]
    mujoco.mj_forward(m, data)
    cam.lookat[:] = data.xpos[1]
    r.update_scene(data, camera=cam)
    img = r.render()
    if label and Image is not None:
        im = Image.fromarray(img); dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, 12 + 9 * len(label), 22], fill=(0, 0, 0))
        dr.text((6, 5), label, fill=(255, 255, 255))
        img = np.asarray(im)
    h, w = img.shape[0] // 16 * 16, img.shape[1] // 16 * 16
    frames.append(img[:h, :w])
imageio.mimsave("%(out)s", frames, fps=%(fps)d)
print("OK")
'''


def _render_trajectory_subprocess(frames_file, save_path, fps, label, backend=None):
    """Render a saved pose trajectory to mp4 in a mujoco-only subprocess (osmesa on a headless
    box, glfw locally). Returns save_path, or None if it could not render."""
    import subprocess
    import sys
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    asset = os.path.join(root, "src", "envs", "assets", "drone.xml")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    backend = backend or ("osmesa" if os.path.isdir("/kaggle") else "glfw")
    child = _RENDER_CHILD % dict(backend=backend, frames=frames_file, asset=asset,
                                 out=save_path, fps=fps, label=label)
    try:
        res = subprocess.run([sys.executable, "-c", child],
                             capture_output=True, text=True, timeout=600)
        if res.returncode == 0 and os.path.exists(save_path):
            return save_path
        print("policy-video render failed:", (res.stderr or "")[-500:])
        return None
    except Exception as e:
        print("policy-video render error:", e)
        return None


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
