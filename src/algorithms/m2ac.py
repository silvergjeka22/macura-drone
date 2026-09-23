"""M2AC — masking the least-trustworthy imagined transitions (Pan et al., 2020).

The alternative uncertainty strategy: instead of TRUNCATING rollouts by length
(MACURA), M2AC rolls out a fixed horizon and then MASKS OUT the most uncertain
imagined transitions, keeping only the most trustworthy fraction, with a reward
penalty proportional to uncertainty.

Same SAC backbone + ensemble as MACURA/MBPO. The per-transition uncertainty is
M2AC's own ONE-VS-REST (OvR) disagreement (`ovr_uncertainty` — Pan et al. 2020):
each ensemble member's prediction scored by the NLL under every OTHER member. Set
`rollout.m2ac.uncertainty = "gjs"` to instead reuse MACURA's GJS signal (a
controlled "masking vs truncation with the SAME uncertainty" ablation).

Pure-function library.
"""

from __future__ import annotations

import numpy as np

from src.models import ensemble as ens
from src.algorithms.macura import compute_gjs, trust_threshold


def ovr_uncertainty(means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    """M2AC one-vs-rest (OvR) disagreement (Pan et al., 2020).

    means, variances: (num_members, batch, obs_dim). For every ordered member pair
    (i, j), i != j, take the Gaussian NLL of member i's mean prediction under member
    j's predictive Gaussian, and average over all pairs -> (batch,). It is large when
    a member's prediction is unlikely under the "rest", i.e. the ensemble disagrees
    relative to its own confidence. (The constant 0.5*D*log(2*pi) offset is dropped:
    it does not affect the mask ranking or a proportional penalty.)
    """
    eps = 1e-12
    E = means.shape[0]
    batch = means.shape[1]
    var = np.maximum(variances, eps)
    total = np.zeros(batch)
    pairs = 0
    for i in range(E):
        for j in range(E):
            if i == j:
                continue
            total += 0.5 * np.sum(np.log(var[j]) + (means[i] - means[j]) ** 2 / var[j], axis=-1)
            pairs += 1
    return total / max(pairs, 1)


def m2ac_rollout(dynamics_model, agent, start_obs: np.ndarray,
                 reward_fn, done_fn, cfg: dict, diag_state: dict = None):
    """Fixed-horizon rollouts; keep the most-trustworthy transitions only.

    `diag_state` (optional, measurement only): if given, also score every imagined step with
    MACURA's GJS uncertainty and trust threshold, and report the share of the data M2AC finally
    trains on (after its masking) that MACURA would have thrown away. Never changes what it keeps."""
    mcfg = cfg["rollout"]["m2ac"]
    t_max = int(mcfg["t_max"])
    mask_fraction = float(mcfg["mask_fraction"])      # fraction KEPT
    penalty = float(mcfg["uncertainty_penalty"])
    unc_kind = mcfg.get("uncertainty", "ovr")         # "ovr" (M2AC, default) or "gjs"

    from src.algorithms.sac import select_actions

    obs = np.array(start_obs, dtype=np.float32)
    alive = np.ones(obs.shape[0], dtype=bool)
    raw = []  # (obs, act, rew, next_obs, done, uncertainty, gjs-for-diagnostic)
    kappa = np.inf

    for t in range(t_max):
        if not alive.any():
            break
        act = select_actions(agent, obs, evaluate=False)   # batched (vectorized)
        means, variances = ens.member_gaussians(dynamics_model, obs, act)
        u = ovr_uncertainty(means, variances) if unc_kind == "ovr" else compute_gjs(means, variances)
        g = compute_gjs(means, variances) if diag_state is not None else u   # measurement only
        if diag_state is not None and t == 0:
            kappa = trust_threshold(diag_state, g, cfg)
        next_obs = ens.predict(dynamics_model, obs, act)
        # score the state the action LANDS in -> r(s',a), identical to the real env, minus the uncertainty penalty
        rew = reward_fn(next_obs, act) - penalty * u
        done = done_fn(next_obs)
        a = alive.copy()
        raw.append((obs[a], act[a], rew[a], next_obs[a], done[a], u[a], g[a]))
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
    diag = {"kept_fraction": n_keep / len(u_all)}
    if diag_state is not None:
        g_all = np.concatenate([r[6] for r in raw])
        diag["untrusted_frac"] = float(np.mean(g_all[keep_idx] >= kappa))
    return transitions, diag


