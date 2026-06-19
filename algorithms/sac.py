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
    add_transition(agent, obs, act, next_obs, reward, done)
    sac_update(agent, num_updates, batch_size) -> metrics
    select_action(agent, obs, evaluate) -> action
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import gymnasium as gym
    from stable_baselines3 import SAC
except ImportError:
    torch = None
    gym = None
    SAC = None


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


def add_transition(agent, obs, act, next_obs, reward, done):
    """Add one transition to the agent's replay buffer (SB3 expects batched)."""
    agent.replay_buffer.add(
        np.asarray(obs, np.float32).reshape(1, -1),
        np.asarray(next_obs, np.float32).reshape(1, -1),
        np.asarray(act, np.float32).reshape(1, -1),
        np.asarray([reward], np.float32),
        np.asarray([done], np.bool_),
        [{}],
    )


def sac_update(agent, num_updates: int, batch_size: int):
    """Run `num_updates` SAC gradient steps from the agent's replay buffer."""
    if agent.replay_buffer.size() < batch_size or num_updates <= 0:
        return {"updates": 0}
    agent.train(gradient_steps=num_updates, batch_size=batch_size)
    return {"updates": num_updates}


def select_action(agent, obs: np.ndarray, evaluate: bool = False) -> np.ndarray:
    """Mean action (eval) or a sample from the squashed Gaussian (train)."""
    action, _ = agent.predict(np.asarray(obs, np.float32), deterministic=evaluate)
    return action
