"""MBPO — fixed truncated-linear rollout schedule (the no-adaptivity baseline).

Same SAC backbone and same probabilistic ensemble as MACURA; the ONLY
difference is the rollout length, which follows a pre-scheduled linear ramp
(Janner et al., 2019) instead of adapting to model uncertainty.

IMPORTANT (fairness): MBPO's result depends on a *tuned* schedule. An untuned
ramp makes MBPO look artificially bad and turns the comparison into a strawman.
The schedule lives in config.py -> ROLLOUT["mbpo"]["rollout_schedule"] and is
scaled so it REACHES its long horizon within the run (else MBPO never over-imagines).

Pure-function library.
"""

from __future__ import annotations

import numpy as np

from src.models import ensemble as ens


def rollout_length(env_step: int, schedule) -> int:
    """Truncated-linear schedule: [min_len, max_len, start_step, end_step].

    Length ramps linearly from min_len to max_len between start_step and
    end_step, then stays at max_len. This is MBPO's classic mechanism.
    """
    min_len, max_len, start, end = schedule
    if env_step <= start:
        return int(min_len)
    if env_step >= end:
        return int(max_len)
    frac = (env_step - start) / max(1, (end - start))
    return int(round(min_len + frac * (max_len - min_len)))


def mbpo_rollout(dynamics_model, agent, start_obs: np.ndarray,
                 reward_fn, done_fn, env_step: int, cfg: dict):
    """Branched rollouts of a FIXED, scheduled length (no uncertainty check)."""
    schedule = cfg["rollout"]["mbpo"]["rollout_schedule"]
    horizon = rollout_length(env_step, schedule)

    from src.algorithms.sac import select_actions

    obs = np.array(start_obs, dtype=np.float32)
    alive = np.ones(obs.shape[0], dtype=bool)
    transitions = []

    for _ in range(horizon):
        if not alive.any():
            break
        act = select_actions(agent, obs, evaluate=False)   # batched (vectorized)
        next_obs = ens.predict(dynamics_model, obs, act)
        rew = reward_fn(obs, act)
        done = done_fn(next_obs)
        if alive.any():
            transitions.append(
                (obs[alive], act[alive], rew[alive], next_obs[alive], done[alive])
            )
        alive = alive & (~done)
        obs = next_obs

    return transitions, {"rollout_length": horizon}


def build_mbpo(obs_dim: int, act_dim: int, cfg: dict, device: str = "cuda"):
    from src.algorithms.sac import build_sac

    agent = build_sac(obs_dim, act_dim, cfg["sac"], device)
    dynamics_model = ens.build_ensemble(cfg["ensemble"], obs_dim, act_dim, device)
    return {"agent": agent, "dynamics_model": dynamics_model}
