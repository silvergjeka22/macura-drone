"""SAC backbone — shared by ALL four algorithms (fairness requirement).

Thin wrapper over mbrl-lib's bundled SAC agent (pytorch_sac_pranz24), so the
exact same learner is used by MACURA, MBPO, M2AC, and the model-free SAC
baseline. Only the data the agent trains on (real vs real+imagined) differs.

Pure-function library.

Public functions:
    build_sac(obs_dim, act_dim, cfg, device) -> agent
    sac_update(agent, replay_buffer, num_updates, cfg) -> update_metrics
    select_action(agent, obs, evaluate) -> action
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    from mbrl.third_party.pytorch_sac_pranz24 import SAC
    from omegaconf import OmegaConf
except ImportError:
    torch = None
    SAC = None
    OmegaConf = None


def build_sac(obs_dim: int, act_dim: int, cfg: dict, device: str = "cuda"):
    """Build the SAC agent. `cfg` is the `sac:` sub-config."""
    if SAC is None:
        raise ImportError("mbrl-lib (pytorch_sac_pranz24) is required for SAC")

    args = OmegaConf.create(
        {
            "gamma": cfg["gamma"],
            "tau": cfg["tau"],
            "alpha": 0.2 if cfg["alpha"] == "auto" else float(cfg["alpha"]),
            "policy": "Gaussian",
            "target_update_interval": cfg["target_update_interval"],
            "automatic_entropy_tuning": cfg["alpha"] == "auto",
            "target_entropy": -float(act_dim),
            "hidden_size": cfg["hidden_size"],
            "lr": cfg["actor_lr"],
            "batch_size": cfg["batch_size"],
            "device": device,
        }
    )
    # SAC expects an action-space-like object with shape; pass dims via a stub.
    action_space = _box(act_dim)
    agent = SAC(obs_dim, action_space, args)
    return agent


def sac_update(agent, replay_buffer, num_updates: int, cfg: dict):
    """Run `num_updates` SAC gradient steps from the given buffer.

    For model-based algorithms `replay_buffer` is the mixed real+model buffer;
    for the SAC baseline it is the environment buffer. Returns mean losses.
    """
    if torch is None:
        raise ImportError("torch required")
    c_losses, p_losses = [], []
    for i in range(num_updates):
        critic_loss, policy_loss, _, _ = agent.update_parameters(
            replay_buffer, cfg["batch_size"], i
        )
        c_losses.append(float(critic_loss))
        p_losses.append(float(policy_loss))
    return {
        "critic_loss": float(np.mean(c_losses)) if c_losses else None,
        "policy_loss": float(np.mean(p_losses)) if p_losses else None,
    }


def select_action(agent, obs: np.ndarray, evaluate: bool = False) -> np.ndarray:
    """Sample (train) or take the mean (eval) action for one observation."""
    return agent.select_action(obs, evaluate=evaluate)


# ── internals ─────────────────────────────────────────────────────────────────
def _box(act_dim: int):
    """Minimal gym-style Box for SAC (actions normalized to [-1, 1])."""
    import gymnasium as gym

    return gym.spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
