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
    """Sweep total thrust and record resulting vertical acceleration proxy —
    a sanity check that actions move the drone as expected."""
    levels = np.linspace(-1, 1, 9)
    final_z = []
    for lv in levels:
        env.reset(seed=0)
        z0 = env.data.qpos[2]
        for _ in range(10):
            env.step(np.full(env.action_space.shape[0], lv, dtype=np.float32))
        final_z.append(float(env.data.qpos[2] - z0))
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
        ax.plot(steps, mean, label=algo.upper())
        ax.fill_between(steps, mean - std, mean + std, alpha=0.2)
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
        ax.plot(steps, mean, label=algo.upper())
        ax.fill_between(steps, mean - std, mean + std, alpha=0.2)
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
    ax.bar(algos, means, yerr=stds, capsize=5)
    ax.set_ylabel("best evaluation return"); ax.set_title("Final policy quality")
    ax.grid(alpha=0.3, axis="y")
    _maybe_save(fig, save_path)
    return fig


# ── internals ─────────────────────────────────────────────────────────────────
def _maybe_save(fig, save_path):
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
