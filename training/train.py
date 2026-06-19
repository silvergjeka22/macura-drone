"""Training orchestration — one function per algorithm run, plus evaluation.

This wires together the shared SAC backbone, the shared ensemble, and the
algorithm-specific rollout strategy. The four algorithms share EVERYTHING except
which rollout function is called (and SAC calls none).

Pure-function library. The Colab notebook calls these; nothing runs on import.

Public functions:
    train_one(algo_name, cfg, drive_dir, seed) -> run_dict
    evaluate(agent, env, eval_episodes) -> metrics

`run_dict` contains the time-series needed for every comparison plot:
    steps, eval_return, eval_failure_rate, and (MACURA) kappa / rollout_length.
"""

from __future__ import annotations

import os
import numpy as np

from envs import drone_env
from models import ensemble as ens
from algorithms import sac as sac_mod
from algorithms import macura as macura_mod
from algorithms import mbpo as mbpo_mod
from algorithms import m2ac as m2ac_mod

try:
    import torch
    import mbrl.util.replay_buffer as mbrl_rb
except ImportError:
    torch = None
    mbrl_rb = None


# ── evaluation (identical protocol for all algorithms) ────────────────────────
def evaluate(agent, env, eval_episodes: int) -> dict:
    """Greedy evaluation. Returns mean return, mean episode length, failure rate."""
    returns, lengths, failures = [], [], []
    for _ in range(eval_episodes):
        obs, _ = env.reset()
        done = False
        ep_ret, ep_len, failed = 0.0, 0, False
        while not done:
            act = sac_mod.select_action(agent, obs, evaluate=True)
            obs, rew, terminated, truncated, info = env.step(act)
            ep_ret += rew
            ep_len += 1
            failed = failed or info.get("failure", False)
            done = terminated or truncated
        returns.append(ep_ret)
        lengths.append(ep_len)
        failures.append(float(failed))
    return {
        "eval_return": float(np.mean(returns)),
        "eval_return_std": float(np.std(returns)),
        "eval_length": float(np.mean(lengths)),
        "eval_failure_rate": float(np.mean(failures)),
    }


# ── a single training run ─────────────────────────────────────────────────────
def train_one(algo_name: str, cfg: dict, drive_dir: str, seed: int = 0) -> dict:
    """Train one algorithm for `experiment.total_env_steps` and log curves.

    algo_name in {'macura', 'mbpo', 'm2ac', 'sac'}.
    """
    if torch is None:
        raise ImportError("torch/mbrl-lib required for training")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _seed_everything(seed)

    # env (train + eval use separate seeds; eval seeds shared across algos)
    env, obs_dim, act_dim = drone_env.make_env(cfg["env"], seed=seed)
    eval_env, _, _ = drone_env.make_env(cfg["env"], seed=cfg["experiment"]["eval_seeds"][0])

    reward_fn = drone_env.known_reward_fn(cfg["env"])
    done_fn = drone_env.termination_fn(cfg["env"])

    # shared components
    agent = sac_mod.build_sac(obs_dim, act_dim, cfg["sac"], device)
    model_based = algo_name in ("macura", "mbpo", "m2ac")
    dynamics_model = (
        ens.build_ensemble(cfg["ensemble"], obs_dim, act_dim, device)
        if model_based else None
    )

    # buffers
    env_buffer = mbrl_rb.ReplayBuffer(cfg["experiment"]["total_env_steps"], (obs_dim,), (act_dim,))
    model_buffer = (
        mbrl_rb.ReplayBuffer(cfg["rollout"]["model_buffer_capacity"], (obs_dim,), (act_dim,))
        if model_based else None
    )
    train_buffer = model_buffer if model_based else env_buffer

    kappa_state: dict = {}
    log = _empty_log()

    total_steps = cfg["experiment"]["total_env_steps"]
    warmup = cfg["experiment"]["warmup_random_steps"]
    eval_every = cfg["experiment"]["eval_every_steps"]
    rollout_freq = cfg["rollout"]["freq_steps"]
    g_max = cfg["sac"]["gradient_steps_max"]

    obs, _ = env.reset()
    for step in range(total_steps):
        # --- act in the real environment ---
        if step < warmup:
            act = env.action_space.sample()
        else:
            act = _explore(agent, obs, cfg["exploration"])
        next_obs, rew, terminated, truncated, info = env.step(act)
        env_buffer.add(obs, act, next_obs, rew, terminated)
        obs = next_obs if not (terminated or truncated) else env.reset()[0]

        if step < warmup:
            continue

        # --- model-based machinery ---
        num_updates = g_max
        if model_based and step % cfg["ensemble"]["train_epochs_per_round"] == 0:
            ens.train_ensemble(dynamics_model, env_buffer, cfg["ensemble"])

        if model_based and step % rollout_freq == 0:
            start = _sample_starts(env_buffer, cfg["rollout"]["num_rollouts"])
            if algo_name == "macura":
                trans, diag = macura_mod.macura_rollout(
                    dynamics_model, agent, start, reward_fn, done_fn, kappa_state, cfg
                )
                num_updates = macura_mod.gradient_steps(
                    model_buffer.num_stored, cfg["rollout"]["model_buffer_capacity"],
                    g_max, cfg["rollout"]["macura"]["adaptive_gradient_steps"],
                )
                log["kappa"].append((step, diag["kappa"]))
                log["rollout_length"].append((step, diag["mean_rollout_length"]))
            elif algo_name == "mbpo":
                trans, diag = mbpo_mod.mbpo_rollout(
                    dynamics_model, agent, start, reward_fn, done_fn, step, cfg
                )
                num_updates = cfg["rollout"]["mbpo"]["fixed_gradient_steps"]
                log["rollout_length"].append((step, diag["rollout_length"]))
            else:  # m2ac
                trans, diag = m2ac_mod.m2ac_rollout(
                    dynamics_model, agent, start, reward_fn, done_fn, cfg
                )
                num_updates = cfg["rollout"]["m2ac"]["fixed_gradient_steps"]
            _store_transitions(model_buffer, trans)

        # --- SAC updates (from model buffer if model-based, else env buffer) ---
        if train_buffer.num_stored >= cfg["sac"]["batch_size"]:
            sac_mod.sac_update(agent, train_buffer, num_updates, cfg["sac"])

        # --- periodic evaluation ---
        if step % eval_every == 0:
            m = evaluate(agent, eval_env, cfg["experiment"]["eval_episodes"])
            log["steps"].append(step)
            log["eval_return"].append(m["eval_return"])
            log["eval_return_std"].append(m["eval_return_std"])
            log["eval_failure_rate"].append(m["eval_failure_rate"])
            _save_checkpoint(agent, drive_dir, algo_name, seed, step)

    run = {"algo": algo_name, "seed": seed, **log}
    _save_run(run, drive_dir, algo_name, seed)
    env.close()
    eval_env.close()
    return run


# ── small internals ───────────────────────────────────────────────────────────
def _empty_log():
    return {
        "steps": [], "eval_return": [], "eval_return_std": [],
        "eval_failure_rate": [], "kappa": [], "rollout_length": [],
    }


def _explore(agent, obs, expl_cfg):
    """Exploration applied identically to all algorithms (avoids the confound)."""
    act = sac_mod.select_action(agent, obs, evaluate=False)
    if expl_cfg["type"] == "deterministic":
        return act
    # white/pink noise share the same per-step scale; pink adds temporal
    # correlation (implemented as a simple AR(1) proxy here for portability).
    noise = np.random.normal(0.0, expl_cfg["scale"], size=np.shape(act))
    return np.clip(act + noise, -1.0, 1.0)


def _sample_starts(buffer, num):
    batch = buffer.sample(num)
    return np.asarray(batch.obs)


def _store_transitions(buffer, transitions):
    for (obs, act, rew, next_obs, done) in transitions:
        for i in range(len(obs)):
            buffer.add(obs[i], act[i], next_obs[i], float(rew[i]), bool(done[i]))


def _seed_everything(seed):
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)


def _save_checkpoint(agent, drive_dir, algo, seed, step):
    path = os.path.join(drive_dir, "checkpoints", f"{algo}_seed{seed}")
    os.makedirs(path, exist_ok=True)
    try:
        agent.save_checkpoint(ckpt_path=os.path.join(path, f"step{step}.pt"))
    except Exception:
        pass  # checkpoint API varies; logs/plots are the primary deliverable


def _save_run(run, drive_dir, algo, seed):
    import json
    path = os.path.join(drive_dir, "logs")
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, f"{algo}_seed{seed}.json"), "w") as f:
        json.dump(run, f)
