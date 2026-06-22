"""Training orchestration — Dyna loop for all four algorithms, plus evaluation.

Backbone: Stable-Baselines3 SAC (shared learner) + our PyTorch probabilistic
ensemble + our pure-function rollout strategies. The four algorithms share
EVERYTHING except which rollout function is called (SAC calls none).

Data flow:
  * a small numpy `_RealBuffer` holds real transitions  -> trains the ensemble
    and supplies rollout start states;
  * the SB3 agent's own replay buffer holds the data the SAC learner trains on
    (REAL transitions for the SAC baseline; MODEL transitions for MACURA/MBPO/M2AC).

Pure-function library. The Colab notebook calls these.

Public functions:
    train_one(algo_name, cfg, drive_dir, seed) -> run_dict
    evaluate(agent, env, eval_episodes)        -> metrics
"""

from __future__ import annotations

import os
import json
import time
import numpy as np

from envs import drone_env
from models import ensemble as ens
from algorithms import sac as sac_mod
from algorithms import macura as macura_mod
from algorithms import mbpo as mbpo_mod
from algorithms import m2ac as m2ac_mod

try:
    import torch
except ImportError:
    torch = None


# ── a minimal numpy replay buffer for REAL data ───────────────────────────────
class _RealBuffer:
    def __init__(self, capacity, obs_dim, act_dim):
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.act = np.zeros((capacity, act_dim), np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), np.float32)
        self.cap, self.size, self.ptr = capacity, 0, 0

    def add(self, obs, act, next_obs):
        i = self.ptr
        self.obs[i], self.act[i], self.next_obs[i] = obs, act, next_obs
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def all(self):
        s = self.size
        return {"obs": self.obs[:s], "act": self.act[:s], "next_obs": self.next_obs[:s]}

    def sample_obs(self, n):
        idx = np.random.randint(0, self.size, size=min(n, self.size))
        return self.obs[idx]


# ── evaluation (identical protocol for all algorithms) ────────────────────────
def evaluate(agent, env, eval_episodes: int) -> dict:
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
    """Train one algorithm in {'macura','mbpo','m2ac','sac'} and log curves."""
    if torch is None:
        raise ImportError("torch required for training")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _seed_everything(seed)

    env, obs_dim, act_dim = drone_env.make_env(cfg["env"], seed=seed)
    eval_env, _, _ = drone_env.make_env(cfg["env"], seed=cfg["experiment"]["eval_seeds"][0])
    reward_fn = drone_env.known_reward_fn(cfg["env"])
    done_fn = drone_env.termination_fn(cfg["env"])

    model_based = algo_name in ("macura", "mbpo", "m2ac")
    buf_cap = cfg["rollout"]["model_buffer_capacity"] if model_based else 1_000_000
    sac_cfg = {**cfg["sac"], "buffer_size": buf_cap}
    agent = sac_mod.build_sac(obs_dim, act_dim, sac_cfg, device, seed)
    dynamics_model = ens.build_ensemble(cfg["ensemble"], obs_dim, act_dim, device) if model_based else None

    total_steps = cfg["experiment"]["total_env_steps"]
    real_buffer = _RealBuffer(total_steps, obs_dim, act_dim)

    warmup = cfg["experiment"]["warmup_random_steps"]
    eval_every = cfg["experiment"]["eval_every_steps"]
    rollout_freq = cfg["rollout"]["freq_steps"]
    num_rollouts = cfg["rollout"]["num_rollouts"]
    g_max = cfg["sac"]["gradient_steps_max"]
    batch = cfg["sac"]["batch_size"]

    kappa_state: dict = {}
    log = _empty_log()
    t0 = time.time()

    obs, _ = env.reset(seed=seed)
    for step in range(total_steps):
        # --- act in the real environment ---
        act = env.action_space.sample() if step < warmup else _explore(agent, obs, cfg["exploration"])
        next_obs, rew, terminated, truncated, info = env.step(act)
        real_buffer.add(obs, act, next_obs)
        if not model_based:
            sac_mod.add_transition(agent, obs, act, next_obs, rew, terminated)
        obs = next_obs if not (terminated or truncated) else env.reset()[0]
        if step < warmup:
            continue

        num_updates = g_max

        # --- model-based: retrain ensemble + generate fresh rollouts ---
        if model_based and step % rollout_freq == 0 and real_buffer.size >= max(num_rollouts, batch):
            ens.train_ensemble(dynamics_model, real_buffer.all(), cfg["ensemble"])
            start = real_buffer.sample_obs(num_rollouts)
            if algo_name == "macura":
                trans, diag = macura_mod.macura_rollout(
                    dynamics_model, agent, start, reward_fn, done_fn, kappa_state, cfg)
                num_updates = macura_mod.gradient_steps(
                    agent.replay_buffer.size(), buf_cap, g_max,
                    cfg["rollout"]["macura"]["adaptive_gradient_steps"])
                log["kappa"].append((step, diag["kappa"]))
                log["rollout_length"].append((step, diag["mean_rollout_length"]))
                log["gjs"].append((step, diag["base_uncertainty"]))
                log["rollout_length_samples"].append((step, diag["lengths"]))
            elif algo_name == "mbpo":
                trans, diag = mbpo_mod.mbpo_rollout(
                    dynamics_model, agent, start, reward_fn, done_fn, step, cfg)
                num_updates = cfg["rollout"]["mbpo"]["fixed_gradient_steps"]
                log["rollout_length"].append((step, diag["rollout_length"]))
            else:  # m2ac
                trans, diag = m2ac_mod.m2ac_rollout(
                    dynamics_model, agent, start, reward_fn, done_fn, cfg)
                num_updates = cfg["rollout"]["m2ac"]["fixed_gradient_steps"]
            _store_model_transitions(agent, trans)

        # --- SAC updates ---
        if agent.replay_buffer.size() >= batch:
            sac_mod.sac_update(agent, num_updates, batch)

        # --- periodic evaluation ---
        if step % eval_every == 0:
            m = evaluate(agent, eval_env, cfg["experiment"]["eval_episodes"])
            log["steps"].append(step)
            log["eval_return"].append(m["eval_return"])
            log["eval_return_std"].append(m["eval_return_std"])
            log["eval_failure_rate"].append(m["eval_failure_rate"])
            log["wall_clock"].append((step, time.time() - t0))
            print(f"[{algo_name} seed{seed}] step {step}  return {m['eval_return']:.1f}"
                  f"  fail {m['eval_failure_rate']:.2f}")

    ckpt = _save_checkpoint(agent, drive_dir, algo_name, seed)
    run = {"algo": algo_name, "seed": seed, "checkpoint": ckpt, **log}
    _save_run(run, drive_dir, algo_name, seed)
    env.close()
    eval_env.close()
    return run


# ── small internals ───────────────────────────────────────────────────────────
def _empty_log():
    return {"steps": [], "eval_return": [], "eval_return_std": [],
            "eval_failure_rate": [], "kappa": [], "rollout_length": [],
            "gjs": [], "rollout_length_samples": [], "wall_clock": []}


def _explore(agent, obs, expl_cfg):
    """Exploration applied identically to all algorithms (avoids the confound)."""
    act = sac_mod.select_action(agent, obs, evaluate=False)
    if expl_cfg["type"] == "deterministic":
        return act
    noise = np.random.normal(0.0, expl_cfg["scale"], size=np.shape(act))
    return np.clip(act + noise, -1.0, 1.0)


def _store_model_transitions(agent, transitions):
    for (obs, act, rew, next_obs, done) in transitions:
        for i in range(len(obs)):
            sac_mod.add_transition(agent, obs[i], act[i], next_obs[i], float(rew[i]), bool(done[i]))


def _seed_everything(seed):
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)


def _save_checkpoint(agent, drive_dir, algo, seed):
    """Save the SB3 agent so the notebook can reload it for video rollouts."""
    path = os.path.join(drive_dir, "checkpoints", f"{algo}_seed{seed}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        agent.save(path)
        return path + ".zip"
    except Exception:
        return None


def _save_run(run, drive_dir, algo, seed):
    path = os.path.join(drive_dir, "logs")
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, f"{algo}_seed{seed}.json"), "w") as f:
        json.dump(run, f)
