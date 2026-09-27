"""MACURA: model-based rollouts truncated where the ensemble disagrees (Frauenknecht et al. 2024).

    compute_gjs      geometric Jensen-Shannon disagreement between members (Eq. 15-19)
    update_kappa     trust threshold kappa = xi x running mean of the zeta-quantile of first-step GJS (Eq. 21)
    macura_rollout   branched rollouts that stop per sample once GJS >= kappa (Algorithm 2)
    gradient_steps   SAC updates per real step scaled by the imagined-data fill level (Eq. 22)

trust_threshold / fast_share are measurements only: the same rule applied to MBPO and M2AC rollouts.
"""

from __future__ import annotations

import numpy as np

from src.models import ensemble as ens

_VZ = 10                                          # obs column of the vertical velocity


def _kl_diag_gaussian(mu_p, var_p, mu_q, var_q):
    eps = 1e-12
    var_p = np.maximum(var_p, eps)
    var_q = np.maximum(var_q, eps)
    term = np.log(var_q / var_p) + (var_p + (mu_p - mu_q) ** 2) / var_q - 1.0
    return 0.5 * np.sum(term, axis=-1)


def _gjs_pair(mu_e, var_e, mu_f, var_f):
    """D_GJS between two diagonal Gaussians via their geometric-mean Gaussian (Eq. 17-19)."""
    eps = 1e-12
    inv_e = 1.0 / np.maximum(var_e, eps)
    inv_f = 1.0 / np.maximum(var_f, eps)
    var_ef = 1.0 / (0.5 * inv_e + 0.5 * inv_f)
    mu_ef = var_ef * (0.5 * inv_e * mu_e + 0.5 * inv_f * mu_f)
    return 0.5 * _kl_diag_gaussian(mu_e, var_e, mu_ef, var_ef) + \
        0.5 * _kl_diag_gaussian(mu_f, var_f, mu_ef, var_ef)


def compute_gjs(means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    """Mean pairwise GJS over members. means, variances: (E, B, D) -> (B,)."""
    E = means.shape[0]
    total = np.zeros(means.shape[1])
    for e in range(E):
        for f in range(e):
            total += _gjs_pair(means[e], variances[e], means[f], variances[f])
    return (2.0 / (E * (E - 1))) * total


def update_kappa(kappa_state: dict, first_step_uncertainties: np.ndarray,
                 zeta: float, xi: float) -> float:
    base = float(np.quantile(first_step_uncertainties, zeta))
    kappa_state["sum"] = kappa_state.get("sum", 0.0) + base
    kappa_state["rounds"] = kappa_state.get("rounds", 0) + 1
    kappa = xi * kappa_state["sum"] / kappa_state["rounds"]
    kappa_state["kappa"] = kappa
    return kappa


def trust_threshold(diag_state: dict, first_step_u: np.ndarray, cfg: dict) -> float:
    """kappa as MACURA would set it, kept in a separate state (measurement on MBPO / M2AC rollouts)."""
    m = cfg["rollout"]["macura"]
    return update_kappa(diag_state, first_step_u, float(m["zeta"]), float(m["xi"]))


def fast_threshold(cfg: dict) -> float:
    """Sink speed (m/s) counted as a fast descent: where the lift loss starts."""
    return float(cfg.get("env", {}).get("vrs_speed", 1.2))


def is_fast(obs: np.ndarray, threshold: float) -> np.ndarray:
    return -np.asarray(obs)[:, _VZ] > threshold


def fast_share(transitions, threshold: float) -> float:
    """Share of the imagined training transitions that start in a fast descent."""
    n = sum(len(t[0]) for t in transitions)
    return (sum(int(is_fast(t[0], threshold).sum()) for t in transitions) / n) if n else float("nan")


def macura_rollout(dynamics_model, agent, start_obs: np.ndarray,
                   reward_fn, done_fn, kappa_state: dict, cfg: dict):
    """Branched rollouts from `start_obs`; each one stops at the first step whose GJS >= kappa.

    Returns (transitions [(obs, act, rew, next_obs, done)], diag)."""
    mcfg = cfg["rollout"]["macura"]
    t_max = int(mcfg["t_max"])
    zeta = float(mcfg["zeta"])
    xi = float(mcfg["xi"])

    obs = np.array(start_obs, dtype=np.float32)
    alive = np.ones(obs.shape[0], dtype=bool)
    lengths = np.zeros(obs.shape[0], dtype=int)
    transitions = []
    kappa = kappa_state.get("kappa", np.inf)
    base_u = 0.0
    v_fast = fast_threshold(cfg)
    zs = {f"{q}_{tag}": 0 for q in ("u", "n", "t") for tag in ("fast", "slow")}

    from src.algorithms.sac import select_actions

    for t in range(t_max):
        if not alive.any():
            break
        act = select_actions(agent, obs, evaluate=False)
        means, variances = ens.member_gaussians(dynamics_model, obs, act)
        u = compute_gjs(means, variances)
        if t == 0:
            base_u = float(np.quantile(u, zeta))
            kappa = update_kappa(kappa_state, u, zeta, xi)

        next_obs = ens.predict(dynamics_model, obs, act)
        rew = reward_fn(next_obs, act)
        done = done_fn(next_obs)
        trust = u < kappa
        keep = alive & trust

        fast = is_fast(obs, v_fast)
        for tag, m in (("fast", alive & fast), ("slow", alive & ~fast)):
            zs["u_" + tag] += float(u[m].sum())
            zs["n_" + tag] += int(m.sum())
            zs["t_" + tag] += int((m & trust).sum())
        if keep.any():
            transitions.append((obs[keep], act[keep], rew[keep], next_obs[keep], done[keep]))
            lengths[keep] += 1

        alive = alive & trust & (~done)
        obs = next_obs

    def ratio(a, b):
        return zs[a] / zs[b] if zs[b] else float("nan")

    n_all = zs["n_fast"] + zs["n_slow"]
    diag = {
        "kappa": float(kappa),
        "mean_rollout_length": float(lengths.mean()),
        "max_rollout_length": int(lengths.max()) if len(lengths) else 0,
        "base_uncertainty": base_u,
        "lengths": lengths.tolist(),
        "unc_fast": ratio("u_fast", "n_fast"), "unc_slow": ratio("u_slow", "n_slow"),
        "trust_fast": ratio("t_fast", "n_fast"), "trust_slow": ratio("t_slow", "n_slow"),
        "fast_frac": fast_share(transitions, v_fast),
        "untrusted_frac": 0.0,
        "discarded_frac": (1.0 - float(lengths.sum()) / n_all) if n_all else float("nan"),
    }
    return transitions, diag


def gradient_steps(model_buffer_size: int, model_buffer_capacity: int,
                   g_max: int, adaptive: bool = True) -> int:
    """Eq. 22: g_max x fill level of the (live) imagined data, at least 1."""
    if not adaptive:
        return g_max
    frac = min(1.0, model_buffer_size / max(1, model_buffer_capacity))
    return max(1, int(round(g_max * frac)))
