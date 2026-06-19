"""M2AC — masking the least-trustworthy imagined transitions (Pan et al., 2020).

The alternative uncertainty strategy: instead of TRUNCATING rollouts by length
(MACURA), M2AC rolls out a fixed horizon and then MASKS OUT the most uncertain
imagined transitions, keeping only the most trustworthy fraction, with a reward
penalty proportional to uncertainty.

Same SAC backbone + ensemble as MACURA/MBPO. We reuse the GJS uncertainty from
algorithms.macura as the per-transition uncertainty signal so the comparison is
"masking vs adaptive truncation" using the SAME uncertainty estimate (the paper
notes M2AC's original OvR estimate is brittle — see tasks.md for the faithful
variant).

Pure-function library.
"""

from __future__ import annotations

import numpy as np

from models import ensemble as ens
from algorithms.macura import compute_gjs


def m2ac_rollout(dynamics_model, agent, start_obs: np.ndarray,
                 reward_fn, done_fn, cfg: dict):
    """Fixed-horizon rollouts; keep the most-trustworthy transitions only."""
    mcfg = cfg["rollout"]["m2ac"]
    t_max = int(mcfg["t_max"])
    mask_fraction = float(mcfg["mask_fraction"])      # fraction KEPT
    penalty = float(mcfg["uncertainty_penalty"])

    from algorithms.sac import select_action

    obs = np.array(start_obs, dtype=np.float32)
    alive = np.ones(obs.shape[0], dtype=bool)
    raw = []  # (obs, act, rew, next_obs, done, uncertainty)

    for _ in range(t_max):
        if not alive.any():
            break
        act = np.stack([select_action(agent, o, evaluate=False) for o in obs])
        means, variances = ens.member_gaussians(dynamics_model, obs, act)
        u = compute_gjs(means, variances)
        next_obs, _ = ens.predict(dynamics_model, obs, act)
        rew = reward_fn(obs, act) - penalty * u       # uncertainty reward penalty
        done = done_fn(next_obs)
        a = alive.copy()
        raw.append((obs[a], act[a], rew[a], next_obs[a], done[a], u[a]))
        alive = alive & (~done)
        obs = next_obs

    if not raw:
        return [], {"kept_fraction": 0.0}

    # pool all transitions, then mask out the least-trustworthy (highest u)
    obs_all = np.concatenate([r[0] for r in raw])
    act_all = np.concatenate([r[1] for r in raw])
    rew_all = np.concatenate([r[2] for r in raw])
    nxt_all = np.concatenate([r[3] for r in raw])
    done_all = np.concatenate([r[4] for r in raw])
    u_all = np.concatenate([r[5] for r in raw])

    n_keep = max(1, int(round(mask_fraction * len(u_all))))
    keep_idx = np.argsort(u_all)[:n_keep]             # smallest uncertainty kept

    transitions = [(
        obs_all[keep_idx], act_all[keep_idx], rew_all[keep_idx],
        nxt_all[keep_idx], done_all[keep_idx],
    )]
    return transitions, {"kept_fraction": n_keep / len(u_all)}


def build_m2ac(obs_dim: int, act_dim: int, cfg: dict, device: str = "cuda"):
    from algorithms.sac import build_sac

    agent = build_sac(obs_dim, act_dim, cfg["sac"], device)
    dynamics_model = ens.build_ensemble(cfg["ensemble"], obs_dim, act_dim, device)
    return {"agent": agent, "dynamics_model": dynamics_model}
