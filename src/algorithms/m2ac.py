"""M2AC — masking the least-trustworthy imagined transitions (Pan et al., 2020).

The alternative uncertainty strategy: instead of TRUNCATING rollouts by length
(MACURA), M2AC rolls out a fixed horizon and then MASKS OUT the most uncertain
imagined transitions, keeping only the most trustworthy fraction, with a reward
penalty proportional to uncertainty.

Same SAC backbone + ensemble as MACURA/MBPO. The per-transition uncertainty is
M2AC's own ONE-VS-REST (OvR) disagreement (`ovr_uncertainty` — Pan et al. 2020):
the KL divergence of each member's prediction from the (moment-matched) rest. Set
`rollout.m2ac.uncertainty = "gjs"` to instead reuse MACURA's GJS signal (a
controlled "masking vs truncation with the SAME uncertainty" ablation).

Pure-function library.
"""

from __future__ import annotations

import numpy as np

from src.models import ensemble as ens
from src.algorithms.macura import compute_gjs, trust_threshold, fast_share, fast_threshold


def ovr_uncertainty(means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    """M2AC one-vs-rest (OvR) disagreement (Pan et al., 2020).

    means, variances: (num_members, batch, obs_dim). For each member i, the "rest" (all other
    members) is summarised by one moment-matched Gaussian, and the uncertainty is
    KL( N_i || N_rest ), averaged over i -> (batch,). A KL divergence is never negative: it is 0
    when the members agree and grows as member i's prediction becomes unlikely under the rest.

    (Before 2026-09-23 this was a pairwise NLL that kept the log-variance term: in the normalised
    delta space that term is strongly negative (~-30), so M2AC's "penalty" r - u was really a ~+30
    BONUS per imagined step - not M2AC.)
    """
    eps = 1e-12
    E = means.shape[0]
    var = np.maximum(variances, eps)
    total = np.zeros(means.shape[1])
    for i in range(E):
        rest = [j for j in range(E) if j != i]
        mu_r = means[rest].mean(axis=0)
        var_r = np.maximum((var[rest] + means[rest] ** 2).mean(axis=0) - mu_r ** 2, eps)
        total += 0.5 * np.sum(np.log(var_r / var[i]) + (var[i] + (means[i] - mu_r) ** 2) / var_r - 1.0,
                              axis=-1)
    return total / E


def m2ac_rollout(dynamics_model, agent, start_obs: np.ndarray,
                 reward_fn, done_fn, cfg: dict, diag_state: dict = None):
    """Fixed-horizon rollouts; keep the most-trustworthy transitions only.

    `diag_state` (optional, measurement only): if given, also score every imagined step with
    MACURA's GJS uncertainty and trust threshold, and report the share of the data M2AC finally
    trains on (after its masking) that MACURA would have thrown away. Never changes what it keeps."""
    mcfg = cfg["rollout"]["m2ac"]
    if mcfg.get("mode", "pooled") == "paper":
        return _m2ac_paper_rollout(dynamics_model, agent, start_obs, reward_fn, done_fn, cfg, diag_state)
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
    diag = {"kept_fraction": n_keep / len(u_all),
            "fast_frac": fast_share(transitions, fast_threshold(cfg))}
    if diag_state is not None:
        g_all = np.concatenate([r[6] for r in raw])
        diag["untrusted_frac"] = float(np.mean(g_all[keep_idx] >= kappa))
    return transitions, diag


def _ovr_chosen(means, variances, pick):
    """Eq. 8-9 of the M2AC paper: u_k = KL( N_k || N_-k ) for the member k that ACTUALLY made each row's
    prediction (N_-k = moment-matched Gaussian of the other members). (B,)"""
    eps = 1e-12
    E, B, _ = means.shape
    var = np.maximum(variances, eps)
    rows = np.arange(B)
    mu_k, var_k = means[pick, rows], var[pick, rows]                       # (B, D)
    s_mu, s_m2 = means.sum(0), (var + means ** 2).sum(0)
    mu_r = (s_mu - mu_k) / (E - 1)
    var_r = np.maximum((s_m2 - (var_k + mu_k ** 2)) / (E - 1) - mu_r ** 2, eps)
    return 0.5 * np.sum(np.log(var_r / var_k) + (var_k + (mu_k - mu_r) ** 2) / var_r - 1.0, axis=-1)


def _m2ac_paper_rollout(dynamics_model, agent, start_obs, reward_fn, done_fn, cfg, diag_state=None):
    """M2AC exactly as Pan et al. (2020) run it (Algorithm 2, NON-STOP mode, Sec. 5.1):
      * at EVERY step h, rank this step's samples by the one-vs-rest KL of the member that predicted them
        and keep only the w_h*B least uncertain, w_h = (H - h) / (2 (H + 1))  (~45% at h=0 -> ~5% at h=H-1,
        25% of all generated steps overall),
      * stored reward r - alpha*u with alpha = 1e-3,
      * every rollout continues to the next step (non-stop) until it terminates or reaches H.
    `diag_state`: measurement only, as in the pooled version."""
    mcfg = cfg["rollout"]["m2ac"]
    H = int(mcfg["t_max"])
    alpha = float(mcfg.get("uncertainty_penalty", 1e-3))
    from src.algorithms.sac import select_actions

    obs = np.array(start_obs, dtype=np.float32)
    alive = np.ones(obs.shape[0], dtype=bool)
    transitions, n_gen, n_keep, kept_g = [], 0, 0, []
    kappa = np.inf
    for h in range(H):
        if not alive.any():
            break
        act = select_actions(agent, obs, evaluate=False)
        means, variances = ens.member_gaussians(dynamics_model, obs, act)
        next_obs, pick = ens.predict(dynamics_model, obs, act, return_pick=True)
        u = _ovr_chosen(means, variances, pick)
        if diag_state is not None:                       # measurement only (no randomness)
            g = compute_gjs(means, variances)
            if h == 0:
                kappa = trust_threshold(diag_state, g, cfg)
        rew = reward_fn(next_obs, act) - alpha * u
        done = done_fn(next_obs)
        idx = np.flatnonzero(alive)
        k = int(np.floor((H - h) / (2.0 * (H + 1)) * len(idx)))
        keep = idx[np.argsort(u[idx])[:k]] if k > 0 else idx[:0]
        n_gen += len(idx); n_keep += len(keep)
        if len(keep):
            transitions.append((obs[keep], act[keep], rew[keep], next_obs[keep], done[keep]))
            if diag_state is not None:
                kept_g.append(g[keep] >= kappa)
        alive = alive & (~done)                          # non-stop: all live rollouts continue
        obs = next_obs
    diag = {"kept_fraction": n_keep / n_gen if n_gen else 0.0,
            "fast_frac": fast_share(transitions, fast_threshold(cfg))}
    if diag_state is not None:
        diag["untrusted_frac"] = float(np.mean(np.concatenate(kept_g))) if kept_g else float("nan")
    return transitions, diag


