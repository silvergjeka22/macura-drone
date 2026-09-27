"""M2AC: fixed-horizon rollouts that keep only the least uncertain imagined steps (Pan et al. 2020).

Paper mode (used by race2, delivery2, race): at step h keep the (H - h) / (2 (H + 1)) share of samples
with the lowest one-vs-rest KL of the member that made the prediction; reward r - alpha * u; rollouts
continue non-stop. The pooled mode (cage, delivery) keeps a fixed share of all steps.
"""

from __future__ import annotations

import numpy as np

from src.models import ensemble as ens
from src.algorithms.macura import compute_gjs, trust_threshold, fast_share, fast_threshold


def ovr_uncertainty(means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    """Mean over members i of KL(N_i || moment-matched N_rest). (E, B, D) -> (B,)."""
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
    """With `diag_state`, also measures the share of the kept data that MACURA's rule would reject."""
    mcfg = cfg["rollout"]["m2ac"]
    if mcfg.get("mode", "pooled") == "paper":
        return _m2ac_paper_rollout(dynamics_model, agent, start_obs, reward_fn, done_fn, cfg, diag_state)
    t_max = int(mcfg["t_max"])
    mask_fraction = float(mcfg["mask_fraction"])
    penalty = float(mcfg["uncertainty_penalty"])
    unc_kind = mcfg.get("uncertainty", "ovr")

    from src.algorithms.sac import select_actions

    obs = np.array(start_obs, dtype=np.float32)
    alive = np.ones(obs.shape[0], dtype=bool)
    raw = []
    kappa = np.inf

    for t in range(t_max):
        if not alive.any():
            break
        act = select_actions(agent, obs, evaluate=False)
        means, variances = ens.member_gaussians(dynamics_model, obs, act)
        u = ovr_uncertainty(means, variances) if unc_kind == "ovr" else compute_gjs(means, variances)
        g = compute_gjs(means, variances) if diag_state is not None else u
        if diag_state is not None and t == 0:
            kappa = trust_threshold(diag_state, g, cfg)
        next_obs = ens.predict(dynamics_model, obs, act)
        rew = reward_fn(next_obs, act) - penalty * u
        done = done_fn(next_obs)
        a = alive.copy()
        raw.append((obs[a], act[a], rew[a], next_obs[a], done[a], u[a], g[a]))
        alive = alive & (~done)
        obs = next_obs

    if not raw:
        return [], {"kept_fraction": 0.0}

    obs_all = np.concatenate([r[0] for r in raw])
    act_all = np.concatenate([r[1] for r in raw])
    rew_all = np.concatenate([r[2] for r in raw])
    nxt_all = np.concatenate([r[3] for r in raw])
    done_all = np.concatenate([r[4] for r in raw])
    u_all = np.concatenate([r[5] for r in raw])

    n_keep = max(1, int(round(mask_fraction * len(u_all))))
    keep_idx = np.argsort(u_all)[:n_keep]

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
    """M2AC Eq. 8-9: KL(N_k || N_-k) for the member k that made each row's prediction. -> (B,)"""
    eps = 1e-12
    E, B, _ = means.shape
    var = np.maximum(variances, eps)
    rows = np.arange(B)
    mu_k, var_k = means[pick, rows], var[pick, rows]
    s_mu, s_m2 = means.sum(0), (var + means ** 2).sum(0)
    mu_r = (s_mu - mu_k) / (E - 1)
    var_r = np.maximum((s_m2 - (var_k + mu_k ** 2)) / (E - 1) - mu_r ** 2, eps)
    return 0.5 * np.sum(np.log(var_r / var_k) + (var_k + (mu_k - mu_r) ** 2) / var_r - 1.0, axis=-1)


def _m2ac_paper_rollout(dynamics_model, agent, start_obs, reward_fn, done_fn, cfg, diag_state=None):
    """Pan et al. (2020) Algorithm 2, non-stop mode: keep share (H - h) / (2 (H + 1)) at step h (25% overall)."""
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
        if diag_state is not None:
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
        alive = alive & (~done)
        obs = next_obs
    diag = {"kept_fraction": n_keep / n_gen if n_gen else 0.0,
            "fast_frac": fast_share(transitions, fast_threshold(cfg))}
    if diag_state is not None:
        diag["untrusted_frac"] = float(np.mean(np.concatenate(kept_g))) if kept_g else float("nan")
    return transitions, diag


