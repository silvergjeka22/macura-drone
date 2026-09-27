"""M2AC: fixed-horizon rollouts that keep only the least uncertain imagined steps (Pan et al. 2020, non-stop mode).

At step h, the (H - h) / (2 (H + 1)) share of samples with the lowest one-vs-rest KL of the member that made the
prediction is kept (25% of all steps overall); stored reward r - alpha * u; every rollout continues to the horizon.
"""

import numpy as np

from src.models import ensemble as ens
from src.algorithms.macura import compute_gjs, trust_threshold, fast_share, fast_threshold


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


def m2ac_rollout(dynamics_model, agent, start_obs, reward_fn, done_fn, cfg, diag_state=None):
    """With `diag_state`, also measures the share of the kept data that MACURA's rule would reject."""
    from src.algorithms.sac import select_actions
    H = int(cfg["rollout"]["m2ac"]["t_max"])
    alpha = float(cfg["rollout"]["m2ac"]["uncertainty_penalty"])

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
        n_gen += len(idx)
        n_keep += len(keep)
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
