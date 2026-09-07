"""SAC backbone — shared by ALL four algorithms (fairness), built on SB3.

We use Stable-Baselines3's SAC as the single learner for MACURA, MBPO, M2AC and
the model-free SAC baseline. Only the data the agent trains on differs:
  * SAC baseline  -> trains from REAL transitions (agent.replay_buffer);
  * model-based   -> trains from MODEL transitions added to agent.replay_buffer.

We drive SB3 manually (add transitions + call .train) instead of .learn(), so
the Dyna loop in training/train.py controls env interaction, rollouts and UTD.

Pure-function library.

Public functions:
    build_sac(obs_dim, act_dim, cfg, device) -> agent
    build_replay_buffer(agent, capacity) -> ReplayBuffer       (a second SB3 buffer)
    add_transition(agent, obs, act, next_obs, reward, done)
    add_to_buffer(buf, obs, act, next_obs, reward, done)
    sac_update(agent, num_updates, batch_size) -> metrics
    sac_update_mixed(agent, real_buf, model_buf, num_updates, batch, rr, rng) -> metrics
    select_action(agent, obs, evaluate) -> action
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import gymnasium as gym
    from stable_baselines3 import SAC
    from stable_baselines3.common.buffers import ReplayBuffer
except ImportError:
    torch = None
    gym = None
    SAC = None
    ReplayBuffer = None


class _DummyEnv(gym.Env if gym is not None else object):
    """Spaces-only env so SB3 can build its networks/buffer without stepping."""

    def __init__(self, obs_dim, act_dim):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (act_dim,), np.float32)

    def reset(self, *, seed=None, options=None):
        return np.zeros(self.observation_space.shape, np.float32), {}

    def step(self, action):
        obs = np.zeros(self.observation_space.shape, np.float32)
        return obs, 0.0, False, False, {}


def build_sac(obs_dim: int, act_dim: int, cfg: dict, device: str = "cuda", seed: int = 0):
    """Build the SB3 SAC agent. `cfg` is the `sac:` sub-config."""
    if SAC is None:
        raise ImportError("stable-baselines3 is required for SAC")
    device = device if torch.cuda.is_available() else "cpu"
    ent_coef = "auto" if cfg["alpha"] == "auto" else float(cfg["alpha"])
    agent = SAC(
        "MlpPolicy",
        _DummyEnv(obs_dim, act_dim),
        learning_rate=cfg["actor_lr"],
        buffer_size=cfg.get("buffer_size", 1_000_000),
        batch_size=cfg["batch_size"],
        tau=cfg["tau"],
        gamma=cfg["gamma"],
        ent_coef=ent_coef,
        target_update_interval=cfg["target_update_interval"],
        learning_starts=0,
        policy_kwargs={"net_arch": [cfg["hidden_size"], cfg["hidden_size"]]},
        device=device,
        seed=seed,
        verbose=0,
    )
    # initialize SB3's internal logger/counters so .train() works without .learn()
    agent._setup_learn(total_timesteps=0, callback=None)
    return agent


def build_replay_buffer(agent, capacity: int):
    """Build a SECOND SB3 ReplayBuffer matching the agent's spaces/device.

    Used by the model-based Dyna loop to keep a separate REAL buffer alongside the agent's own
    (model/imagined) buffer, so the SAC update can draw whole batches from one or the other
    (batch-level real/imagined mixing — see `sac_update_mixed`). `handle_timeout_termination`
    is off (the loop stores raw terminated flags) to match `add_to_buffer` / `add_transition`.
    """
    if ReplayBuffer is None:
        raise ImportError("stable-baselines3 is required for the replay buffer")
    return ReplayBuffer(
        int(capacity),
        agent.observation_space,
        agent.action_space,
        device=agent.device,
        n_envs=1,
        optimize_memory_usage=False,
        handle_timeout_termination=False,
    )


def add_to_buffer(buf, obs, act, next_obs, reward, done):
    """Add one transition to a given SB3 ReplayBuffer (SB3 expects batched)."""
    buf.add(
        np.asarray(obs, np.float32).reshape(1, -1),
        np.asarray(next_obs, np.float32).reshape(1, -1),
        np.asarray(act, np.float32).reshape(1, -1),
        np.asarray([reward], np.float32),
        np.asarray([done], np.bool_),
        [{}],
    )


def add_transition(agent, obs, act, next_obs, reward, done):
    """Add one transition to the agent's own replay buffer."""
    add_to_buffer(agent.replay_buffer, obs, act, next_obs, reward, done)


def sac_update(agent, num_updates: int, batch_size: int):
    """Run `num_updates` SAC gradient steps from the agent's replay buffer."""
    if agent.replay_buffer.size() < batch_size or num_updates <= 0:
        return {"updates": 0}
    agent.train(gradient_steps=num_updates, batch_size=batch_size)
    return {"updates": num_updates}


class _MixedReplaySampler:
    """Presents SB3's ReplayBuffer.sample() API but returns a WITHIN-BATCH mix: each
    sampled batch is `real_ratio * batch` REAL transitions concatenated with the rest
    MODEL (imagined) transitions. This is the canonical MBPO/MACURA Dyna mix (Janner
    2019): every SAC gradient step sees a steady fraction of real data. Any other
    attribute access is proxied to a real SB3 buffer, so `agent.train()` is untouched."""

    def __init__(self, real_buf, model_buf, real_ratio):
        self.real_buf = real_buf
        self.model_buf = model_buf
        self.real_ratio = float(real_ratio)

    def sample(self, batch_size, env=None):
        from stable_baselines3.common.type_aliases import ReplayBufferSamples
        n_real = int(round(self.real_ratio * batch_size))
        if self.real_buf.size() == 0:
            n_real = 0
        if self.model_buf.size() == 0:
            n_real = batch_size
        n_model = batch_size - n_real
        parts = []
        if n_real > 0:
            parts.append(self.real_buf.sample(n_real, env=env))
        if n_model > 0:
            parts.append(self.model_buf.sample(n_model, env=env))
        if len(parts) == 1:
            return parts[0]
        return ReplayBufferSamples(*(torch.cat([getattr(p, f) for p in parts], dim=0)
                                     for f in ReplayBufferSamples._fields))

    def __getattr__(self, name):                       # proxy everything else to a real buffer
        if name in ("real_buf", "model_buf", "real_ratio"):
            raise AttributeError(name)
        return getattr(self.model_buf, name)


def sac_update_mixed(agent, real_buf, model_buf, num_updates: int, batch_size: int,
                     real_ratio: float, rng=None) -> dict:
    """WITHIN-BATCH real/imagined mixing for the model-based agents (canonical MBPO/MACURA;
    Janner 2019). Every SAC batch is `real_ratio * batch_size` REAL transitions + the rest
    MODEL (imagined) — the standard Dyna mix, so each gradient step gets a steady real-data
    correction (not the higher-variance whole-batch-real-or-model scheme). Implemented by
    pointing the learner at a `_MixedReplaySampler` for the duration; the SAC math is SB3's
    own `agent.train()`, untouched. `rng` is accepted for signature compatibility (unused —
    the mix is a fixed per-batch proportion). The agent's own buffer is restored afterwards.
    """
    n_real = int(round(float(real_ratio) * batch_size))
    if num_updates <= 0:
        return {"updates": 0, "n_real_per_batch": n_real,
                "n_model_per_batch": batch_size - n_real,
                "real_pct": 100.0 * float(real_ratio),
                "imagined_pct": 100.0 * (1.0 - float(real_ratio))}
    saved = agent.replay_buffer
    agent.replay_buffer = _MixedReplaySampler(real_buf, model_buf, real_ratio)
    try:
        agent.train(gradient_steps=int(num_updates), batch_size=batch_size)
    finally:
        agent.replay_buffer = saved
    return {
        "updates": int(num_updates),
        "n_real_per_batch": n_real,
        "n_model_per_batch": batch_size - n_real,
        "real_pct":     100.0 * float(real_ratio),
        "imagined_pct": 100.0 * (1.0 - float(real_ratio)),
    }


def select_action(agent, obs: np.ndarray, evaluate: bool = False) -> np.ndarray:
    """Mean action (eval) or a sample from the squashed Gaussian (train)."""
    action, _ = agent.predict(np.asarray(obs, np.float32), deterministic=evaluate)
    return action


def select_actions(agent, obs_batch: np.ndarray, evaluate: bool = False) -> np.ndarray:
    """Batched action selection — one SB3 `predict` over (N, obs_dim) instead of N
    Python calls. Used to vectorize branched model rollouts (big speedup)."""
    actions, _ = agent.predict(np.asarray(obs_batch, np.float32), deterministic=evaluate)
    return actions
