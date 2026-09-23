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
from src.algorithms.macura import compute_gjs, trust_threshold, fast_share, fast_threshold  # diagnostics


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
                 reward_fn, done_fn, env_step: int, cfg: dict, diag_state: dict = None):
    """Branched rollouts of a FIXED, scheduled length (no uncertainty check).

    `diag_state` (optional, measurement only): if given, also score every imagined step with
    MACURA's GJS uncertainty and trust threshold, and report the share of the data MBPO trains
    on that MACURA would have thrown away. It never changes which transitions MBPO keeps."""
    schedule = cfg["rollout"]["mbpo"]["rollout_schedule"]
    horizon = rollout_length(env_step, schedule)

    from src.algorithms.sac import select_actions

    obs = np.array(start_obs, dtype=np.float32)
    alive = np.ones(obs.shape[0], dtype=bool)
    transitions = []
    n_stored = n_untrusted = 0
    kappa = np.inf

    for t in range(horizon):
        if not alive.any():
            break
        act = select_actions(agent, obs, evaluate=False)   # batched (vectorized)
        if diag_state is not None:                         # measurement only (no randomness)
            u = compute_gjs(*ens.member_gaussians(dynamics_model, obs, act))
            if t == 0:
                kappa = trust_threshold(diag_state, u, cfg)
            n_stored += int(alive.sum())
            n_untrusted += int((alive & (u >= kappa)).sum())
        next_obs = ens.predict(dynamics_model, obs, act)
        rew = reward_fn(next_obs, act)   # score the state the action LANDS in -> r(s',a), identical to the real env
        done = done_fn(next_obs)
        if alive.any():
            transitions.append(
                (obs[alive], act[alive], rew[alive], next_obs[alive], done[alive])
            )
        alive = alive & (~done)
        obs = next_obs

    diag = {"rollout_length": horizon, "fast_frac": fast_share(transitions, fast_threshold(cfg))}
    if diag_state is not None:
        diag["untrusted_frac"] = n_untrusted / n_stored if n_stored else float("nan")
    return transitions, diag


