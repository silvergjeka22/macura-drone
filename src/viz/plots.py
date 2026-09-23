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


def plot_trust_by_zone(macura_runs, save_path=None):
    """Where does MACURA stop trusting its model? Left: mean model uncertainty (GJS) of imagined states
    near the pad (inside `turb_radius`, i.e. the ring + gap) vs in transit. Right: fraction of imagined
    steps MACURA kept (trusted) in each region. Lower trust near the pad = MACURA is cutting exactly the
    imagined landings a fixed-horizon method (MBPO) would still train on."""
    run_list = macura_runs if isinstance(macura_runs, list) else [macura_runs]

    def _nan_series(key):
        series = [np.asarray(r[key], dtype=float) for r in run_list if r.get(key)]
        if not series:
            return None, None
        n = min(len(s) for s in series)
        steps = series[0][:n, 0]
        vals = np.stack([s[:n, 1] for s in series])
        import warnings
        with warnings.catch_warnings():            # rounds with no near-pad states are NaN
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return steps, np.nanmean(vals, axis=0)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    panels = ((ax[0], ("unc_near", "unc_far"), "model uncertainty (GJS)",
               "How unsure the model is"),
              (ax[1], ("trust_near", "trust_far"), "fraction of imagined steps trusted",
               "Where MACURA trusts its model"))
    for a, (kn, kf), ylab, title in panels:
        for key, color, lab in ((kn, "#d62728", "near the pad (ring + gap)"),
                                (kf, "#1f77b4", "transit")):
            steps, m = _nan_series(key)
            if steps is not None:
                a.plot(steps, m, color=color, lw=2, label=lab)
        a.set_xlabel("real environment steps"); a.set_ylabel(ylab); a.set_title(title)
        a.grid(alpha=0.3); a.legend()
    ax[1].set_ylim(-0.02, 1.02)
    fig.suptitle("MACURA: uncertainty and trust, near the pad vs in transit", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
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

    fig.suptitle("MACURA vs MBPO vs M2AC vs SAC -- drone ring landing", fontsize=14)
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
def record_policy_video(ckpt_path, cfg, save_path, seconds=60, seed=999, label=None,
                        device="cpu", n_candidates=60):
    """Save a ~`seconds` mp4 of a trained policy's BEST episodes, safe on a headless GPU box.

    Step 1 (here, no graphics): roll the deterministic policy over `n_candidates` full episodes,
    scoring each. Step 2: rank them (successful LANDINGS first, then no-crash, then higher return,
    then longer) and stitch the best ones back-to-back until the clip is ~`seconds` long, with a
    per-episode caption burned in. Step 3: render in a mujoco-only subprocess. Returns save_path
    or None. Showing the best episodes is honest for a DEMO (it is a highlight reel, clearly the
    strongest runs) - the scientific claim still comes from the aggregate plots, not the video."""
    import tempfile
    try:
        eps = _policy_episodes(ckpt_path, cfg["env"], int(n_candidates), seed, device)
    except Exception as e:
        print("policy rollout failed:", e)
        return None
    if not eps:
        return None
    fps = eps[0]["fps"]
    # rank best-first: landed > not-crashed > higher return > longer (steadier)
    eps.sort(key=lambda e: (e["reached"], not e["crashed"], e["ret"], e["len"]), reverse=True)
    target = int(seconds * fps)
    qpos, mocap, labels = [], [], []
    shown = 0
    n_land = sum(e["reached"] for e in eps)
    for e in eps:
        if len(qpos) >= target and shown >= 3:
            break
        shown += 1
        outcome = "LANDED ✓" if e["reached"] else ("crash ✗" if e["crashed"] else "hover")
        tag = f"{(label + ' | ') if label else ''}clip {shown}: {outcome}  (return {e['ret']:.0f})"
        for k in range(e["len"]):
            qpos.append(e["qpos"][k]); mocap.append(e["mocap"][k]); labels.append(tag)
    print(f"  {label or 'policy'}: {n_land}/{len(eps)} episodes landed; showing best {shown} "
          f"({len(qpos)} frames ~ {len(qpos)/fps:.0f}s)")
    frames_file = tempfile.mktemp(suffix=".npz")
    np.savez(frames_file, qpos=np.array(qpos), mocap=np.array(mocap),
             labels=np.array(labels, dtype=object))
    return _render_trajectory_subprocess(frames_file, save_path, int(fps), label or "")


def _policy_episodes(ckpt_path, env_cfg, n_episodes, seed, device):
    """Run the trained policy (deterministic) for `n_episodes` FULL episodes; return a list of
    per-episode dicts {qpos, mocap, ret, len, reached, crashed, fps}. No graphics here, so SB3 and
    MuJoCo coexist safely. Each episode uses a distinct seed for variety."""
    from stable_baselines3 import SAC
    from src.envs import drone_env
    env, _, _ = drone_env.make_env(env_cfg, seed=seed, render=False)
    agent = SAC.load(ckpt_path, device=device)
    fps = int(round(1.0 / (env.model.opt.timestep * env.action_repeat)))
    eps = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i)
        qpos, mocap = [], []
        ret = 0.0; reached = False; crashed = False; done = False
        while not done:
            act = agent.predict(np.asarray(obs, np.float32), deterministic=True)[0]
            obs, r, term, trunc, info = env.step(act)
            ret += float(r)
            qpos.append(np.array(env.data.qpos, np.float64))
            mocap.append(np.array(env.data.mocap_pos, np.float64))
            reached = reached or info.get("reached", False)
            crashed = crashed or info.get("failure", False)
            done = term or trunc
        eps.append({"qpos": np.array(qpos), "mocap": np.array(mocap), "ret": ret,
                    "len": len(qpos), "reached": reached, "crashed": crashed, "fps": fps})
    env.close()
    return eps


# child script: renders a saved pose trajectory. Imports ONLY mujoco (never torch/SB3), so
# software graphics can never crash the training kernel.
_RENDER_CHILD = '''
import os, sys
os.environ["MUJOCO_GL"] = "%(backend)s"
import numpy as np, mujoco, imageio
d = np.load("%(frames)s", allow_pickle=True)
qpos, mocap = d["qpos"], d["mocap"]
labels = d["labels"] if "labels" in d.files else None
fallback = "%(label)s"
m = mujoco.MjModel.from_xml_path("%(asset)s")
data = mujoco.MjData(m)
r = mujoco.Renderer(m, height=480, width=640)
cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.distance, cam.elevation, cam.azimuth = 5.2, -22.0, 90.0
lookat = None
try:
    from PIL import Image, ImageDraw
except Exception:
    Image = None
# stream frames to disk (do NOT hold a ~1-min clip in memory)
writer = imageio.get_writer("%(out)s", fps=%(fps)d, macro_block_size=16)
for i in range(len(qpos)):
    data.qpos[:] = qpos[i]
    if mocap.shape[1] > 0:
        data.mocap_pos[:] = mocap[i]
    mujoco.mj_forward(m, data)
    tgt = data.xpos[1]
    lookat = tgt.copy() if lookat is None else 0.90 * lookat + 0.10 * tgt   # smooth follow
    cam.lookat[:] = lookat
    r.update_scene(data, camera=cam)
    img = r.render()
    txt = str(labels[i]) if labels is not None else fallback
    if txt and Image is not None:
        im = Image.fromarray(img); dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, 12 + 8 * len(txt), 22], fill=(0, 0, 0))
        dr.text((6, 5), txt, fill=(255, 255, 255))
        img = np.asarray(im)
    h, w = img.shape[0] // 16 * 16, img.shape[1] // 16 * 16
    writer.append_data(np.ascontiguousarray(img[:h, :w]))
writer.close()
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


def stitch_videos_grid(video_paths: dict, save_path: str, fps=30, cols=2, downscale=2,
                       max_frames=3600):
    """Side-by-side reel: tile the given {label: mp4_path} clips into one grid video with each label
    burned in. STREAMS frame-by-frame (readers as iterators; shorter clips hold their last frame),
    so a 4x1-min grid never loads every frame into memory at once."""
    import imageio
    readers, names, iters, last = [], [], [], []
    for name, path in video_paths.items():
        rd = imageio.get_reader(path)
        readers.append(rd); names.append(name); iters.append(iter(rd)); last.append(None)
    if not readers:
        raise ValueError("no readable clips in video_paths")

    def _next(k):
        try:
            f = np.asarray(next(iters[k]))[::downscale, ::downscale, :3]
            last[k] = f
            return f, True
        except StopIteration:
            return last[k], False

    first = [_next(k)[0] for k in range(len(readers))]
    h = max(f.shape[0] for f in first if f is not None)
    w = max(f.shape[1] for f in first if f is not None)
    rows = int(np.ceil(len(readers) / cols))
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

    def _tile(f, name):
        out = np.zeros((h, w, 3), np.uint8)
        if f is not None:
            out[: f.shape[0], : f.shape[1]] = f
        return _label(out, name.upper())

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    writer = imageio.get_writer(save_path, fps=fps, macro_block_size=16)
    blank = np.zeros((h, w, 3), np.uint8)
    cur = first
    t = 0
    while t < max_frames:
        tiles = [_tile(cur[k], names[k]) for k in range(len(readers))]
        tiles += [blank] * (rows * cols - len(tiles))
        grid = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])
        writer.append_data(_pad16(grid))
        t += 1
        nxt, alive = [], False
        for k in range(len(readers)):
            f, a = _next(k)
            nxt.append(f); alive = alive or a
        cur = nxt
        if not alive:                      # every clip has ended
            break
    writer.close()
    for rd in readers:
        rd.close()
    return save_path


def _maybe_save(fig, save_path):
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=130, bbox_inches="tight")


# ── environment explainers (how the ring task is built) ───────────────────────
def plot_env_layout(cfg, seeds=(0, 1, 2, 3), save_path=None):
    """Top-down MAP of the ring task for a few resets: the pad (green) with its landing tolerance,
    the no-fly columns (red) that RING it leaving one entry gap, and the drone start (blue) outside.
    Shows the drone must thread the gap and land in the middle."""
    import matplotlib.patches as mpatches
    from src.envs import drone_env
    ec = cfg["env"]
    env, _, _ = drone_env.make_env(ec, seed=0, render=False)
    orad = env.obs_radius
    lr = float(ec.get("land_radius", 0.5))
    seeds = list(seeds)
    fig, axes = plt.subplots(1, len(seeds), figsize=(3.7 * len(seeds), 3.9))
    if len(seeds) == 1:
        axes = [axes]
    for ax, sd in zip(axes, seeds):
        env.reset(seed=sd)
        pad = env._target[:2]
        ax.add_patch(mpatches.Circle((0, 0), 0.13, color="#1f77b4", zorder=3))
        ax.annotate("start", (0, 0), color="#1f77b4", fontsize=8, ha="center", va="top",
                    xytext=(0, -0.3), textcoords="data")
        tr = float(ec.get("turb_radius", 0.0))
        zone_on = any(float(ec.get(k, 0.0)) > 0.0
                      for k in ("turb_force", "wake_gamma", "pad_downwash", "ge_gain"))
        if zone_on and tr > 0.0:                                 # the hard-to-predict landing zone
            ax.add_patch(mpatches.Circle(pad, tr, color="#ff7f0e", alpha=0.13, zorder=0))
            ax.add_patch(mpatches.Circle(pad, tr, fill=False, ls=":", color="#ff7f0e", lw=1.2, zorder=0))
        ax.add_patch(mpatches.Circle(pad, lr, color="#2ca02c", alpha=0.20, zorder=1))
        ax.add_patch(mpatches.Circle(pad, 0.09, color="#2ca02c", zorder=3))
        ax.annotate("pad", pad, color="#2ca02c", fontsize=8, ha="center",
                    xytext=(pad[0], pad[1] + 0.28), textcoords="data")
        for o in env._obstacles:
            ax.add_patch(mpatches.Circle(o, orad, color="#d62728", alpha=0.55, zorder=2))
        ax.plot([0, pad[0]], [0, pad[1]], "k--", lw=0.9, alpha=0.5, zorder=0)  # approach
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_title(f"reset seed {sd}", fontsize=10)
        ax.set_xlim(-2.7, 2.7); ax.set_ylim(-2.7, 2.7)
    env.close()
    fig.suptitle("Ring task (top-down): calm transit, then thread the gap into the hard-to-predict "
                 "landing zone (orange) and land on the pad", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _maybe_save(fig, save_path)
    return fig


def plot_wind_process(cfg, steps=250, seed=0, save_path=None):
    """The process noise the learned model has to cope with (it lives only in the real physics).
    Uniform config (landing-zone physics off): OU gust traces + gust strength vs hover thrust.
    Landing-zone config: gusts / wake flow map, disturbance vs distance to the pad, ground effect."""
    ec = cfg["env"]
    wf = float(ec.get("wind_force", 0.0)); wc = float(ec.get("wind_correlation", 0.95))
    tf = float(ec.get("turb_force", 0.0)); tc = float(ec.get("turb_correlation", 0.8))
    tr = float(ec.get("turb_radius", 1.3)); ramp = float(ec.get("turb_ramp", 0.15))
    an0 = float(ec.get("action_noise", 0.0)); an1 = float(ec.get("action_noise_zone", an0))
    geg = float(ec.get("ge_gain", 0.0)); ges = float(ec.get("ge_scale", 0.25))

    wake = float(ec.get("wake_gamma", 0.0)) > 0.0 or float(ec.get("pad_downwash", 0.0)) > 0.0
    drag = float(ec.get("air_drag", 0.8))
    act = float(ec.get("actuator_noise", 0.0))

    if not (tf > 0 or wake or max(an0, an1) > 0 or geg > 0):
        # UNIFORM process noise (the landing-zone physics is off): OU gusts + actuator noise everywhere
        hover = 4.3
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
        for sd in (seed, seed + 1, seed + 2):
            rng = np.random.default_rng(sd); w = 0.0; trace = []
            for _ in range(steps):
                w = wc * w + np.sqrt(1 - wc ** 2) * rng.normal(0, wf); trace.append(w)
            ax[0].plot(trace, lw=1.3, alpha=0.85, label=f"gust sample {sd}")
        ax[0].axhspan(-wf, wf, color="0.6", alpha=0.15, label=f"±1 std ({wf} N)")
        ax[0].axhline(0, color="k", lw=0.5)
        ax[0].set_xlabel("control step"); ax[0].set_ylabel("gust force, one axis (N)")
        ax[0].set_title(f"Wind: smooth OU gusts (correlation {wc})"); ax[0].legend(fontsize=8)
        ax[0].grid(alpha=0.3)
        rng = np.random.default_rng(seed); w = np.zeros(3); mags = []
        for _ in range(20000):
            w = wc * w + np.sqrt(1 - wc ** 2) * rng.normal(0, wf, 3); mags.append(np.linalg.norm(w))
        ax[1].hist(100 * np.array(mags) / hover, bins=40, color="#1f77b4", edgecolor="white")
        ax[1].set_xlabel("gust force as % of hover thrust"); ax[1].set_ylabel("count")
        ax[1].set_title(f"Gust strength (+ {act*100:.0f}% random thrust noise per rotor)")
        ax[1].grid(alpha=0.3, axis="y")
        fig.suptitle("Process noise: wind gusts + actuator noise, the same everywhere "
                     "(hover thrust ≈ 4.3 N)", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        _maybe_save(fig, save_path)
        return fig

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    if wake:
        # top-down map of the deterministic air flow around the ring (column wake + pad downwash)
        import matplotlib.patches as mpatches
        from src.envs import drone_env
        env, _, _ = drone_env.make_env(ec, seed=0, render=False)
        env.reset(seed=seed)
        pad = env._target[:2]
        g = np.linspace(-1.6, 1.6, 23)
        X, Y = np.meshgrid(pad[0] + g, pad[1] + g)
        U = np.zeros_like(X); Vv = np.zeros_like(X); W = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                v = env.air_velocity(np.array([X[i, j], Y[i, j], 0.5]))
                U[i, j], Vv[i, j], W[i, j] = v
        sp = np.hypot(U, Vv)
        q = ax[0].quiver(X, Y, U, Vv, sp, cmap="plasma", scale=25, width=0.004)
        fig.colorbar(q, ax=ax[0], label="horizontal air speed (m/s)")
        for o in env._obstacles:
            ax[0].add_patch(mpatches.Circle(o, env.obs_radius, color="#d62728", alpha=0.6))
        ax[0].add_patch(mpatches.Circle(pad, 0.09, color="#2ca02c"))
        ax[0].set_aspect("equal")
        ax[0].set_title("Air flow around the ring (deterministic wake)")
        ax[0].set_xlabel("x (m)"); ax[0].set_ylabel("y (m)")
        # radial profile: mean horizontal air speed and |downwash| at each distance from the pad
        d = np.linspace(0, 3.0, 120)
        ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        hs, vs = [], []
        for r in d:
            vv = [env.air_velocity(np.array([pad[0] + r * np.cos(a), pad[1] + r * np.sin(a), 0.5]))
                  for a in ang]
            hs.append(np.mean([np.hypot(v[0], v[1]) for v in vv])); vs.append(np.mean([abs(v[2]) for v in vv]))
        env.close()
        ax[1].plot(d, drag * np.array(hs), color="#ff7f0e", lw=2, label="wake force, mean (N)")
        ax[1].plot(d, drag * np.array(vs), color="#8c564b", lw=2, label="downwash force, mean (N)")
    else:
        rng = np.random.default_rng(seed)
        w = t = 0.0; calm, zone = [], []
        for _ in range(steps):
            w = wc * w + np.sqrt(1 - wc ** 2) * rng.normal(0, wf)
            t = tc * t + np.sqrt(1 - tc ** 2) * rng.normal(0, tf) if tf > 0 else 0.0
            calm.append(w); zone.append(w + t)
        ax[0].plot(zone, color="#ff7f0e", lw=1.2, label=f"landing zone (+turbulence {tf} N)")
        ax[0].plot(calm, color="#1f77b4", lw=1.6, label=f"calm transit ({wf} N)")
        ax[0].axhline(0, color="k", lw=0.5)
        ax[0].set_xlabel("control step"); ax[0].set_ylabel("gust force, one axis (N)")
        ax[0].set_title("Gusts: calm in transit, choppy near the pad"); ax[0].legend(fontsize=8)
        ax[0].grid(alpha=0.3)
        d = np.linspace(0, 3.0, 200)

    s = 1.0 / (1.0 + np.exp((d - tr) / max(ramp, 1e-6)))
    ax[1].plot(d, np.sqrt(wf ** 2 + (s * tf) ** 2), color="#1f77b4", lw=2, label="random gust std (N)")
    if max(an0, an1) > 0:
        ax[1].plot(d, an0 + s * (an1 - an0), color="#9467bd", lw=2, label="hidden action noise σ")
    ax[1].axvline(tr, ls=":", color="k", lw=1, label=f"landing-zone edge ({tr} m)")
    ax[1].set_xlabel("horizontal distance to pad (m)"); ax[1].set_ylabel("disturbance strength")
    ax[1].set_title("Disturbance vs distance to the pad"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)

    h = np.linspace(0, 2.0, 200)
    ax[2].plot(h, 100 * geg * np.exp(-h / max(ges, 1e-6)), color="#2ca02c", lw=2)
    ax[2].axvline(float(ec.get("pad_z", 0.2)), ls=":", color="k", lw=1, label="landing height")
    ax[2].set_xlabel("height above ground (m)"); ax[2].set_ylabel("extra lift (%)")
    ax[2].set_title("Ground effect: extra lift only near the ground"); ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)

    fig.suptitle("Process noise: the model is reliable in calm transit, unreliable in the landing zone "
                 "(hover thrust ≈ 4.3 N)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _maybe_save(fig, save_path)
    return fig


def plot_reward_landscape(cfg, save_path=None):
    """How the reward SHAPES behaviour: (left) the per-step reward pulling the drone toward the pad;
    (right) the 'settle well' - reward is high only when the drone is BOTH close AND slow (= a real
    landing), which is what drives `reach`. Uses the exact env reward, so this is what the agent sees."""
    from src.envs.drone_env import _drone_reward
    ec = cfg["env"]; rw = ec["reward"]
    n_obs = int(ec.get("n_obstacles", 4))
    orad = float(ec.get("obstacle_radius", 0.3))
    osc = float(rw.get("obs_scale", 0.5))
    dim = 14 + 2 * n_obs

    def obs_at(d, s):
        o = np.zeros(dim)
        o[0] = d                        # pad-relative x = horizontal distance
        o[3] = float(ec.get("pad_z", 0.2))
        o[4] = 1.0                      # qw = 1 (upright)
        o[8] = s                        # vx = speed
        o[14:14 + 2 * n_obs] = 5.0      # obstacles far away
        return o

    D = np.linspace(0, 3, 140)
    r0 = np.array([_drone_reward(obs_at(d, 0.0), np.zeros(4), rw, n_obs, orad, osc)[0] for d in D])
    S = np.linspace(0, 2, 90)
    Z = np.array([[_drone_reward(obs_at(d, s), np.zeros(4), rw, n_obs, orad, osc)[0] for d in D] for s in S])

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax[0].plot(D, r0, color="#1f77b4", lw=2)
    ax[0].axvline(float(ec.get("land_radius", 0.5)), ls="--", color="#2ca02c", label="land_radius")
    ax[0].set_xlabel("distance to pad (m)"); ax[0].set_ylabel("per-step reward (upright, speed 0)")
    ax[0].set_title("Reward pulls the drone toward the pad"); ax[0].legend(); ax[0].grid(alpha=0.3)
    im = ax[1].pcolormesh(D, S, Z, shading="auto", cmap="viridis")
    fig.colorbar(im, ax=ax[1], label="per-step reward")
    ax[1].set_xlabel("distance to pad (m)"); ax[1].set_ylabel("speed (m/s)")
    ax[1].set_title("The 'settle well': high reward only when CLOSE and SLOW (= land)")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_env_difficulty(cfg, episodes=40, seed=0, save_path=None):
    """How hard the RAW env is: outcomes and episode lengths under an untrained RANDOM policy.
    Most episodes crash/expire and none land - that is the gap the learned policies must close."""
    from src.envs import drone_env
    env, _, ad = drone_env.make_env(cfg["env"], seed=seed, render=False)
    rng = np.random.default_rng(seed)
    lens = []; outcome = {"reached": 0, "crash": 0, "timeout": 0}
    for ep in range(episodes):
        obs, _ = env.reset(seed=2000 + ep)
        done = False; L = 0; reached = False; crashed = False
        while not done:
            a = np.clip(rng.normal(0, 0.5, size=ad), -1, 1).astype(np.float32)
            obs, _, te, tr, info = env.step(a); L += 1
            reached = reached or info.get("reached", False)
            crashed = crashed or info.get("failure", False)
            done = te or tr
        lens.append(L)
        outcome["reached" if reached else ("crash" if crashed else "timeout")] += 1
    env.close()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].bar(list(outcome), list(outcome.values()),
              color=["#2ca02c", "#d62728", "#7f7f7f"])
    ax[0].set_ylabel("episodes"); ax[0].set_title(f"Outcomes under a RANDOM policy ({episodes} eps)")
    ax[1].hist(lens, bins=18, color="#1f77b4", edgecolor="white")
    ax[1].set_xlabel("episode length (control steps)"); ax[1].set_ylabel("count")
    ax[1].set_title("Episode length under a random policy"); ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig
