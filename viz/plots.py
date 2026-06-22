"""Plotting — environment study figures + the four-algorithm comparison figures.

Pure-function library: every function takes data/handles and returns a
matplotlib Figure (and optionally saves it). The notebook decides what to show.

Two groups:
  A. ENVIRONMENT STUDY (notebook part 1, runs with only the env installed):
        plot_rollout_traces, plot_state_distributions, plot_reward_landscape,
        render_frame, plot_action_response.
  B. ALGORITHM COMPARISON (after training):
        plot_sample_efficiency, plot_rollout_depth, plot_failure_rate,
        plot_final_quality.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

from envs import drone_env


# consistent per-algorithm colors across every figure in the notebook
ALGO_COLORS = {
    "macura": "#1f77b4",   # blue  — the proposed method
    "mbpo": "#d62728",     # red   — fixed schedule (expected to wobble)
    "m2ac": "#2ca02c",     # green — masking
    "sac": "#7f7f7f",      # gray  — model-free reference
}


def _color(algo):
    return ALGO_COLORS.get(algo.lower(), None)


# ══════════════════════════════════════════════════════════════════════════════
# A. ENVIRONMENT STUDY
# ══════════════════════════════════════════════════════════════════════════════
def collect_rollout(env, policy="random", num_steps=250, seed=0):
    """Run one episode and record per-step diagnostics. `policy` in
    {'random', 'hover', callable}. Returns a dict of arrays."""
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    rec = {"t": [], "z": [], "pos_err": [], "tilt": [], "reward": [], "action": []}
    for t in range(num_steps):
        if policy == "random":
            act = env.action_space.sample()
        elif policy == "hover":                 # constant mid-thrust (no control)
            act = np.zeros(env.action_space.shape[0], dtype=np.float32)
        elif callable(policy):
            act = policy(obs)
        else:
            raise ValueError(policy)
        obs, rew, terminated, truncated, info = env.step(act)
        rec["t"].append(t)
        rec["z"].append(float(obs[2] + env._current_target()[2]))
        rec["pos_err"].append(info["pos_error"])
        rec["tilt"].append(info["tilt"])
        rec["reward"].append(rew)
        rec["action"].append(np.asarray(act))
        if terminated or truncated:
            break
    return {k: np.asarray(v) for k, v in rec.items()}


def plot_rollout_traces(records: dict, title="Drone rollout", save_path=None):
    """Altitude / position-error / tilt / reward over time for one episode."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 6))
    axes[0, 0].plot(records["t"], records["z"]); axes[0, 0].set_title("Altitude z (m)")
    axes[0, 1].plot(records["t"], records["pos_err"]); axes[0, 1].set_title("Position error (m)")
    axes[1, 0].plot(records["t"], np.degrees(records["tilt"])); axes[1, 0].set_title("Tilt (deg)")
    axes[1, 1].plot(records["t"], records["reward"]); axes[1, 1].set_title("Reward")
    for ax in axes.flat:
        ax.set_xlabel("step"); ax.grid(alpha=0.3)
    fig.suptitle(title); fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_state_distributions(env, num_episodes=20, num_steps=100, save_path=None):
    """Histograms of states visited under a random policy — shows the regime the
    ensemble must learn and where instability lives."""
    z, tilt, perr = [], [], []
    for ep in range(num_episodes):
        rec = collect_rollout(env, policy="random", num_steps=num_steps, seed=ep)
        z += rec["z"].tolist(); tilt += rec["tilt"].tolist(); perr += rec["pos_err"].tolist()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    axes[0].hist(z, bins=40); axes[0].set_title("Altitude z (m)")
    axes[1].hist(np.degrees(tilt), bins=40); axes[1].set_title("Tilt (deg)")
    axes[2].hist(perr, bins=40); axes[2].set_title("Position error (m)")
    for ax in axes: ax.grid(alpha=0.3)
    fig.suptitle("State distribution under random policy"); fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_reward_landscape(cfg, save_path=None):
    """1-D slice of the dense reward vs position error (everything else nominal).
    Demonstrates the reward is smooth and peaked at the target (model-friendly)."""
    reward_fn = drone_env.known_reward_fn(cfg["env"])
    errors = np.linspace(0.0, 1.5, 200)
    obs = np.zeros((len(errors), 13))
    obs[:, 0] = errors            # x position error
    obs[:, 3] = 1.0               # identity quaternion (w=1)
    act = np.zeros((len(errors), 4))
    rewards = reward_fn(obs, act)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(errors, rewards)
    ax.set_xlabel("position error (m)"); ax.set_ylabel("dense reward")
    ax.set_title("Reward landscape (smooth, peaked at target)"); ax.grid(alpha=0.3)
    _maybe_save(fig, save_path)
    return fig


def render_frame(env, save_path=None):
    """Render a single RGB frame of the drone (env must be render=True)."""
    frame = env.render()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.imshow(frame); ax.axis("off"); ax.set_title("Skydio X2 (MuJoCo)")
    _maybe_save(fig, save_path)
    return fig


def plot_action_response(env, save_path=None):
    """Sweep total thrust and record resulting change in altitude — a sanity
    check that actions move the drone as expected. Backend-agnostic: reads the
    altitude from the observation (obs[2] = z - target_z), so it works on both
    the MuJoCo and PyBullet envs."""
    levels = np.linspace(-1, 1, 9)
    final_z = []
    for lv in levels:
        obs, _ = env.reset(seed=0)
        z0 = float(obs[2])
        for _ in range(10):
            obs, *_ = env.step(np.full(env.action_space.shape[0], lv, dtype=np.float32))
        final_z.append(float(obs[2]) - z0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(levels, final_z, "o-")
    ax.set_xlabel("normalized thrust command"); ax.set_ylabel("Δz after 10 steps (m)")
    ax.set_title("Action response (thrust → climb)"); ax.grid(alpha=0.3)
    _maybe_save(fig, save_path)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# B. ALGORITHM COMPARISON  (runs is a list of run_dicts from train_one)
# ══════════════════════════════════════════════════════════════════════════════
def _group_by_algo(runs):
    by = {}
    for r in runs:
        by.setdefault(r["algo"], []).append(r)
    return by


def _mean_std_curve(run_list, ykey):
    """Align runs on shared steps and return (steps, mean, std)."""
    steps = np.asarray(run_list[0]["steps"])
    ys = np.array([r[ykey][: len(steps)] for r in run_list])
    return steps, ys.mean(axis=0), ys.std(axis=0)


def plot_sample_efficiency(runs, save_path=None):
    """Return vs real env steps, mean ± std over seeds, all four algorithms."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for algo, run_list in _group_by_algo(runs).items():
        steps, mean, std = _mean_std_curve(run_list, "eval_return")
        ax.plot(steps, mean, label=algo.upper(), color=_color(algo))
        ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color=_color(algo))
    ax.set_xlabel("real environment steps"); ax.set_ylabel("evaluation return")
    ax.set_title("Sample efficiency"); ax.legend(); ax.grid(alpha=0.3)
    _maybe_save(fig, save_path)
    return fig


def plot_rollout_depth(macura_run, save_path=None):
    """MACURA signature figure: kappa and mean rollout length over training."""
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


def plot_failure_rate(runs, save_path=None):
    """Fraction of evaluation episodes that crash, over training."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for algo, run_list in _group_by_algo(runs).items():
        steps, mean, std = _mean_std_curve(run_list, "eval_failure_rate")
        ax.plot(steps, mean, label=algo.upper(), color=_color(algo))
        ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color=_color(algo))
    ax.set_xlabel("real environment steps"); ax.set_ylabel("failure rate")
    ax.set_title("Failure rate while learning"); ax.legend(); ax.grid(alpha=0.3)
    _maybe_save(fig, save_path)
    return fig


def plot_final_quality(runs, save_path=None):
    """Bar chart of best evaluation return per algorithm (mean ± std over seeds)."""
    by = _group_by_algo(runs)
    algos, means, stds = [], [], []
    for algo, run_list in by.items():
        bests = [max(r["eval_return"]) for r in run_list]
        algos.append(algo.upper()); means.append(np.mean(bests)); stds.append(np.std(bests))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(algos, means, yerr=stds, capsize=5,
           color=[ALGO_COLORS.get(a.lower()) for a in algos])
    ax.set_ylabel("best evaluation return"); ax.set_title("Final policy quality")
    ax.grid(alpha=0.3, axis="y")
    _maybe_save(fig, save_path)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# C. CONCEPTUAL / ILLUSTRATIVE FIGURES  (Part 1 & 3 — synthetic, stand alone)
# ══════════════════════════════════════════════════════════════════════════════
def plot_model_error_growth(save_path=None):
    """Illustrative (synthetic): how prediction error compounds with rollout
    horizon for a FIXED long horizon vs MACURA's uncertainty-adaptive cutoff.
    Motivates 'model exploitation' (paper Sec. 1)."""
    h = np.arange(0, 11)
    err = 0.02 * (1.6 ** h)                 # error compounds geometrically
    cutoff = 5                              # where adaptive truncation would stop
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(h, err, "o-", color="#d62728", label="error under fixed long horizon")
    ax.plot(h[: cutoff + 1], err[: cutoff + 1], "o-", color="#1f77b4",
            label="MACURA truncates here (uncertainty rises)")
    ax.axvline(cutoff, ls="--", color="#1f77b4", alpha=0.6)
    ax.set_xlabel("rollout horizon (steps)"); ax.set_ylabel("model prediction error")
    ax.set_title("Illustrative: model error compounds with horizon")
    ax.legend(); ax.grid(alpha=0.3)
    _maybe_save(fig, save_path)
    return fig


def plot_ensemble_toy_1d(ensemble_cfg, device="cpu", save_path=None):
    """Train a small REAL ensemble (models/ensemble.py) on a 1-D toy function and
    plot every member's prediction. Members agree where data exists and fan out
    out-of-distribution — the visual basis of epistemic uncertainty (Sec. 2.2)."""
    from models import ensemble as ens
    rng = np.random.default_rng(0)
    cfg = {**ensemble_cfg, "num_members": 5, "train_epochs_per_round": 120}
    e = ens.build_ensemble(cfg, 1, 1, device)
    f = lambda x: np.sin(1.5 * x)
    xtr = rng.uniform(-3, 3, (400, 1)).astype("float32")
    atr = np.zeros((400, 1), "float32")
    nxt = (xtr + f(xtr)).astype("float32")          # delta = f(x)
    ens.train_ensemble(e, {"obs": xtr, "act": atr, "next_obs": nxt}, cfg)
    xg = np.linspace(-6, 6, 300).reshape(-1, 1).astype("float32")
    means, _ = ens.member_gaussians(e, xg, np.zeros((300, 1), "float32"))  # (E,B,1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axvspan(-3, 3, color="#cccccc", alpha=0.3, label="training region")
    for m in range(means.shape[0]):
        ax.plot(xg[:, 0], means[m, :, 0], color="#1f77b4", alpha=0.5,
                label="ensemble members" if m == 0 else None)
    ax.plot(xg[:, 0], f(xg[:, 0]), "k--", label="ground truth f(x)=sin(1.5x)")
    ax.set_xlabel("state x"); ax.set_ylabel("predicted delta")
    ax.set_title("Ensemble agrees in-distribution, fans out OOD")
    ax.legend(); ax.grid(alpha=0.3)
    _maybe_save(fig, save_path)
    return fig


def plot_gjs_vs_spread(save_path=None):
    """Synthetic: GJS uncertainty (compute_gjs, Eq. 15-19) as ensemble members'
    means spread apart. Shows what 'high uncertainty' numerically means."""
    from algorithms.macura import compute_gjs
    spreads = np.linspace(0.0, 3.0, 40)
    u = []
    for s in spreads:
        # 5 members, 1-D: means spaced by s, unit variance
        means = (np.linspace(-1, 1, 5) * s).reshape(5, 1, 1)
        var = np.ones((5, 1, 1))
        u.append(float(compute_gjs(means, var)[0]))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(spreads, u, color="#1f77b4")
    ax.set_xlabel("spread of ensemble member means")
    ax.set_ylabel("u_GJS uncertainty")
    ax.set_title("GJS uncertainty grows with member disagreement")
    ax.grid(alpha=0.3)
    _maybe_save(fig, save_path)
    return fig


def plot_rollout_truncation_cartoon(save_path=None):
    """Synthetic cartoon of one rollout: GJS rises along the horizon and crosses
    the threshold kappa; MACURA truncates at the crossing (Algorithm 2)."""
    t = np.arange(0, 11)
    gjs = 0.3 * np.exp(0.35 * t)
    kappa = 3.0
    cross = int(np.argmax(gjs > kappa))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(t, gjs, "o-", color="#1f77b4", label="u_GJS along rollout")
    ax.axhline(kappa, ls="--", color="#d62728", label="threshold κ")
    ax.axvline(cross, ls=":", color="k", alpha=0.6)
    ax.fill_between(t, 0, gjs, where=(t <= cross), color="#1f77b4", alpha=0.1,
                    label="transitions kept")
    ax.set_xlabel("rollout step"); ax.set_ylabel("uncertainty")
    ax.set_title(f"Adaptive truncation: stop at step {cross} (GJS > κ)")
    ax.legend(); ax.grid(alpha=0.3)
    _maybe_save(fig, save_path)
    return fig


def plot_noise_comparison(n=2000, seed=0, save_path=None):
    """Pink (1/f, temporally correlated) vs white (uncorrelated) exploration
    noise: time series + power spectral density. The paper uses pink noise for
    MACURA (Sec. 6.2 / 7.3); this project applies ONE scheme to all four."""
    rng = np.random.default_rng(seed)
    white = rng.normal(size=n)
    f = np.fft.rfftfreq(n)
    f[0] = f[1]
    spec = np.fft.rfft(rng.normal(size=n)) / np.sqrt(f)   # 1/f amplitude
    pink = np.fft.irfft(spec, n=n)
    pink /= pink.std() + 1e-9

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(white[:300], color="#7f7f7f", alpha=0.8, label="white")
    axes[0].plot(pink[:300], color="#1f77b4", alpha=0.9, label="pink (1/f)")
    axes[0].set_title("Noise time series"); axes[0].set_xlabel("step")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    for sig, c, lab in [(white, "#7f7f7f", "white"), (pink, "#1f77b4", "pink")]:
        ps = np.abs(np.fft.rfft(sig)) ** 2
        axes[1].loglog(f[1:], ps[1:], color=c, alpha=0.8, label=lab)
    axes[1].set_title("Power spectral density (pink ∝ 1/f)")
    axes[1].set_xlabel("frequency"); axes[1].set_ylabel("power")
    axes[1].legend(); axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_rollout_strategy_schematic(save_path=None):
    """Schematic of rollout length vs training step for each strategy:
    MACURA adaptive, MBPO fixed truncated-linear, M2AC fixed+mask, SAC none."""
    steps = np.linspace(0, 30000, 100)
    macura = 1 + 8 * (1 - np.exp(-steps / 8000)) + 0.4 * np.sin(steps / 2000)
    mbpo = np.clip(1 + (steps / 15000) * 14, 1, 15)        # truncated-linear ramp
    m2ac = np.full_like(steps, 5.0)                         # fixed length (then masked)
    sac = np.zeros_like(steps)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(steps, macura, color=ALGO_COLORS["macura"], label="MACURA (adaptive)")
    ax.plot(steps, mbpo, color=ALGO_COLORS["mbpo"], label="MBPO (fixed schedule)")
    ax.plot(steps, m2ac, color=ALGO_COLORS["m2ac"], label="M2AC (fixed + mask)")
    ax.plot(steps, sac, color=ALGO_COLORS["sac"], label="SAC (no rollouts)")
    ax.set_xlabel("training step"); ax.set_ylabel("rollout length")
    ax.set_title("Schematic: rollout length by strategy")
    ax.legend(); ax.grid(alpha=0.3)
    _maybe_save(fig, save_path)
    return fig


def plot_reward_heatmap_2d(cfg, save_path=None):
    """2-D heatmap of the dense reward over (x, y) position error (level, still).
    Smooth and peaked at the target ⇒ model-friendly (project reward design)."""
    reward_fn = drone_env.known_reward_fn(cfg["env"])
    g = np.linspace(-1.2, 1.2, 120)
    xx, yy = np.meshgrid(g, g)
    obs = np.zeros((xx.size, 13), np.float32)
    obs[:, 0] = xx.ravel(); obs[:, 1] = yy.ravel(); obs[:, 3] = 1.0  # quat w=1
    r = reward_fn(obs, np.zeros((xx.size, 4), np.float32)).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.pcolormesh(xx, yy, r, shading="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, label="dense reward")
    ax.set_xlabel("x error (m)"); ax.set_ylabel("y error (m)")
    ax.set_title("Reward landscape (2-D, peaked at target)")
    _maybe_save(fig, save_path)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# D. EXTRA RESULT FIGURES  (Part 5 — consume run_dicts from train_one)
# ══════════════════════════════════════════════════════════════════════════════
def plot_gjs_over_training(macura_run, save_path=None):
    """(5.5) Mean first-step GJS (ζ-quantile) vs step — decreases as the ensemble
    learns the dynamics."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if macura_run.get("gjs"):
        s, g = zip(*macura_run["gjs"])
        ax.plot(s, g, color=ALGO_COLORS["macura"])
    ax.set_xlabel("real environment steps"); ax.set_ylabel("first-step u_GJS (ζ-quantile)")
    ax.set_title("MACURA: model uncertainty falls during training")
    ax.grid(alpha=0.3)
    _maybe_save(fig, save_path)
    return fig


def plot_rollout_length_hist(macura_run, save_path=None):
    """(5.6) Histogram of per-rollout lengths early vs late in training — makes
    the adaptivity visible (short when uncertain, longer once reliable)."""
    samples = macura_run.get("rollout_length_samples", [])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if samples:
        early = np.asarray(samples[0][1])
        late = np.asarray(samples[-1][1])
        bins = np.arange(0, max(early.max(), late.max()) + 2) - 0.5
        ax.hist(early, bins=bins, alpha=0.6, label=f"early (step {samples[0][0]})",
                color="#9ecae1", density=True)
        ax.hist(late, bins=bins, alpha=0.6, label=f"late (step {samples[-1][0]})",
                color="#1f77b4", density=True)
        ax.legend()
    ax.set_xlabel("rollout length"); ax.set_ylabel("density")
    ax.set_title("MACURA rollout-length distribution: early vs late")
    ax.grid(alpha=0.3)
    _maybe_save(fig, save_path)
    return fig


def plot_steps_to_target(runs, target_return, save_path=None):
    """(5.7) Real env steps each algorithm needs to first reach `target_return`."""
    by = _group_by_algo(runs)
    algos, steps_needed = [], []
    for algo, run_list in by.items():
        per_seed = []
        for r in run_list:
            steps = np.asarray(r["steps"]); ret = np.asarray(r["eval_return"])
            hit = np.where(ret >= target_return)[0]
            per_seed.append(steps[hit[0]] if len(hit) else np.nan)
        algos.append(algo.upper()); steps_needed.append(np.nanmean(per_seed))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(algos, steps_needed, color=[ALGO_COLORS.get(a.lower()) for a in algos])
    ax.set_ylabel("real steps to reach target")
    ax.set_title(f"Sample efficiency to return ≥ {target_return:g}")
    ax.grid(alpha=0.3, axis="y")
    _maybe_save(fig, save_path)
    return fig


def plot_sample_efficiency_spaghetti(runs, save_path=None):
    """(5.8) Per-seed curves (faint) behind the mean — honest variance."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for algo, run_list in _group_by_algo(runs).items():
        c = _color(algo)
        for r in run_list:
            ax.plot(r["steps"], r["eval_return"], color=c, alpha=0.25, lw=1)
        steps, mean, _ = _mean_std_curve(run_list, "eval_return")
        ax.plot(steps, mean, color=c, lw=2.5, label=algo.upper())
    ax.set_xlabel("real environment steps"); ax.set_ylabel("evaluation return")
    ax.set_title("Sample efficiency (per-seed + mean)")
    ax.legend(); ax.grid(alpha=0.3)
    _maybe_save(fig, save_path)
    return fig


def plot_return_vs_wallclock(runs, save_path=None):
    """(5.9) Return vs wall-clock seconds — the Colab compute caveat (README §7):
    MBRL is sample-efficient but compute-heavy."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for algo, run_list in _group_by_algo(runs).items():
        r = run_list[0]
        if r.get("wall_clock"):
            _, secs = zip(*r["wall_clock"])
            ax.plot(secs, r["eval_return"][: len(secs)], color=_color(algo),
                    label=algo.upper())
    ax.set_xlabel("wall-clock seconds"); ax.set_ylabel("evaluation return")
    ax.set_title("Return vs wall-clock (compute is the Colab bottleneck)")
    ax.legend(); ax.grid(alpha=0.3)
    _maybe_save(fig, save_path)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# E. POLICY VIDEO  (Part 5b — render a trained agent flying)
# ══════════════════════════════════════════════════════════════════════════════
def record_policy_video(agent, env, num_steps=250, save_path=None, fps=30,
                        also_gif=False):
    """Run one deterministic evaluation episode, capturing MuJoCo frames, and
    encode to mp4 (and optionally gif). Requires env created with render=True and
    MUJOCO_GL=egl on the Colab GPU. Returns the saved path.

    The agent is any object with `.predict(obs, deterministic=True)` (SB3 SAC).
    """
    import imageio
    obs, _ = env.reset()
    frames = []
    for _ in range(num_steps):
        action, _ = agent.predict(np.asarray(obs, np.float32), deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        frames.append(env.render())
        if terminated or truncated:
            break
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        imageio.mimsave(save_path, frames, fps=fps)
        if also_gif:
            imageio.mimsave(save_path.replace(".mp4", ".gif"), frames, fps=fps)
    return save_path


# ── internals ─────────────────────────────────────────────────────────────────
def _maybe_save(fig, save_path):
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
