"""MBPO: branched rollouts of a scheduled length, no uncertainty check (Janner et al. 2019)."""

from __future__ import annotations

import numpy as np

from src.models import ensemble as ens
from src.algorithms.macura import compute_gjs, trust_threshold, fast_share, fast_threshold


def rollout_length(env_step: int, schedule) -> int:
    """Truncated-linear ramp [min_len, max_len, start_step, end_step]."""
    min_len, max_len, start, end = schedule
    if env_step <= start:
        return int(min_len)
    if env_step >= end:
        return int(max_len)
    frac = (env_step - start) / max(1, (end - start))
    return int(round(min_len + frac * (max_len - min_len)))


def mbpo_rollout(dynamics_model, agent, start_obs: np.ndarray,
                 reward_fn, done_fn, env_step: int, cfg: dict, diag_state: dict = None):
    """With `diag_state`, also measures the share of MBPO's data that MACURA's rule would reject
    (no effect on what MBPO keeps)."""
    horizon = rollout_length(env_step, cfg["rollout"]["mbpo"]["rollout_schedule"])

    from src.algorithms.sac import select_actions

    obs = np.array(start_obs, dtype=np.float32)
    alive = np.ones(obs.shape[0], dtype=bool)
    transitions = []
    n_stored = n_untrusted = 0
    kappa = np.inf

    for t in range(horizon):
        if not alive.any():
            break
        act = select_actions(agent, obs, evaluate=False)
        if diag_state is not None:
            u = compute_gjs(*ens.member_gaussians(dynamics_model, obs, act))
            if t == 0:
                kappa = trust_threshold(diag_state, u, cfg)
            n_stored += int(alive.sum())
            n_untrusted += int((alive & (u >= kappa)).sum())
        next_obs = ens.predict(dynamics_model, obs, act)
        rew = reward_fn(next_obs, act)
        done = done_fn(next_obs)
        if alive.any():
            transitions.append((obs[alive], act[alive], rew[alive], next_obs[alive], done[alive]))
        alive = alive & (~done)
        obs = next_obs

    diag = {"rollout_length": horizon, "fast_frac": fast_share(transitions, fast_threshold(cfg))}
    if diag_state is not None:
        diag["untrusted_frac"] = n_untrusted / n_stored if n_stored else float("nan")
    return transitions, diag
