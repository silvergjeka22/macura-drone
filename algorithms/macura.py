"""MACURA — uncertainty-aware adaptive rollout truncation (the proposed method).

This is the heart of the project. The rollout differs from MBPO only in WHEN it
stops: a branched rollout continues while the model's epistemic uncertainty
u_GJS stays below an adaptive threshold kappa, and truncates the moment it
exceeds it (Algorithm 2 of the paper).

We implement:
  * `compute_gjs(...)`  — the geometric Jensen-Shannon divergence uncertainty
                          measure (paper Eq. 15-19), closed form for diagonal
                          Gaussians. This is exact, pure math.
  * `update_kappa(...)` — the self-tuning threshold (paper Eq. 21): the running
                          mean of (xi * zeta-quantile of first-step uncertainties).
  * `macura_rollout(...)`— branched model-based rollouts with per-step truncation.
  * `gradient_steps(...)`— update-to-data scaling (paper Eq. 22).

Pure-function library. Uses the shared ensemble (models.ensemble) and the known
analytic reward / termination (envs.drone_env) for fairness.
"""

from __future__ import annotations

import numpy as np

from models import ensemble as ens


# ── uncertainty: geometric Jensen-Shannon divergence (Eq. 15-19) ──────────────
def _kl_diag_gaussian(mu_p, var_p, mu_q, var_q):
    """KL( N(mu_p, var_p) || N(mu_q, var_q) ) for DIAGONAL covariances.

    All inputs broadcast over the last axis (the state dimension).
    """
    eps = 1e-12
    var_p = np.maximum(var_p, eps)
    var_q = np.maximum(var_q, eps)
    term = np.log(var_q / var_p) + (var_p + (mu_p - mu_q) ** 2) / var_q - 1.0
    return 0.5 * np.sum(term, axis=-1)


def _gjs_pair(mu_e, var_e, mu_f, var_f):
    """D_GJS(N_e || N_f) for diagonal Gaussians (geometric mean, alpha=0.5).

    Eq. 17-19: build the geometric-mean Gaussian N_ef, then
      D_GJS = 0.5 KL(N_e || N_ef) + 0.5 KL(N_f || N_ef).
    """
    eps = 1e-12
    inv_e = 1.0 / np.maximum(var_e, eps)
    inv_f = 1.0 / np.maximum(var_f, eps)
    var_ef = 1.0 / (0.5 * inv_e + 0.5 * inv_f)          # Eq. 18 (diagonal)
    mu_ef = var_ef * (0.5 * inv_e * mu_e + 0.5 * inv_f * mu_f)  # Eq. 19
    return 0.5 * _kl_diag_gaussian(mu_e, var_e, mu_ef, var_ef) + \
        0.5 * _kl_diag_gaussian(mu_f, var_f, mu_ef, var_ef)


def compute_gjs(means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    """u_GJS uncertainty (Eq. 15) averaged over all member pairs.

    means, variances: (num_members, batch, obs_dim).
    Returns: (batch,) per-sample uncertainty.
    """
    E = means.shape[0]
    batch = means.shape[1]
    total = np.zeros(batch)
    for e in range(E):
        for f in range(e):  # symmetry: only e > f pairs
            total += _gjs_pair(means[e], variances[e], means[f], variances[f])
    return (2.0 / (E * (E - 1))) * total


# ── adaptive threshold kappa (Eq. 21) ─────────────────────────────────────────
def update_kappa(kappa_state: dict, first_step_uncertainties: np.ndarray,
                 zeta: float, xi: float) -> float:
    """Update kappa with this round's first-step uncertainties.

    base = zeta-quantile of the M first-step uncertainties (upper bound on
    "certain"); kappa = xi * running-mean(base) over rounds.
    `kappa_state` is mutated in place; returns the current kappa.
    """
    base = float(np.quantile(first_step_uncertainties, zeta))
    kappa_state["sum"] = kappa_state.get("sum", 0.0) + base
    kappa_state["rounds"] = kappa_state.get("rounds", 0) + 1
    kappa = xi * kappa_state["sum"] / kappa_state["rounds"]
    kappa_state["kappa"] = kappa
    return kappa


# ── branched rollouts with adaptive truncation (Algorithm 2) ──────────────────
def macura_rollout(dynamics_model, agent, start_obs: np.ndarray,
                   reward_fn, done_fn, kappa_state: dict, cfg: dict):
    """Generate M branched model-based rollouts, truncating per uncertainty.

    Args:
        start_obs : (M, obs_dim) start states sampled from the env buffer.
        reward_fn : known analytic reward(obs, action) (fairness).
        done_fn   : termination(obs) matching the real env envelope.
        kappa_state: dict carrying the running kappa statistics across rounds.
        cfg       : merged config with `rollout.macura` and `sac` fields.

    Returns:
        transitions: list of (obs, act, rew, next_obs, done) arrays.
        diag: dict with 'kappa', 'mean_rollout_length' for the signature plot.
    """
    mcfg = cfg["rollout"]["macura"]
    t_max = int(mcfg["t_max"])
    zeta = float(mcfg["zeta"])
    xi = float(mcfg["xi"])

    obs = np.array(start_obs, dtype=np.float32)
    alive = np.ones(obs.shape[0], dtype=bool)
    lengths = np.zeros(obs.shape[0], dtype=int)
    transitions = []
    kappa = kappa_state.get("kappa", np.inf)

    from algorithms.sac import select_action

    for t in range(t_max):
        if not alive.any():
            break
        act = np.stack([select_action(agent, o, evaluate=False) for o in obs])

        # epistemic uncertainty BEFORE committing the transition
        means, variances = ens.member_gaussians(dynamics_model, obs, act)
        u = compute_gjs(means, variances)

        if t == 0:
            kappa = update_kappa(kappa_state, u, zeta, xi)

        next_obs, _ = ens.predict(dynamics_model, obs, act)
        rew = reward_fn(obs, act)
        done = done_fn(next_obs)

        # keep transitions only where (still alive) AND (uncertainty < kappa)
        trust = u < kappa
        keep = alive & trust
        if keep.any():
            transitions.append(
                (obs[keep], act[keep], rew[keep], next_obs[keep], done[keep])
            )
            lengths[keep] += 1

        # a rollout dies if it left the trust region or terminated
        alive = alive & trust & (~done)
        obs = next_obs

    diag = {
        "kappa": float(kappa),
        "mean_rollout_length": float(lengths.mean()),
        "max_rollout_length": int(lengths.max()) if len(lengths) else 0,
    }
    return transitions, diag


# ── update-to-data scaling (Eq. 22) ───────────────────────────────────────────
def gradient_steps(model_buffer_size: int, model_buffer_capacity: int,
                   g_max: int, adaptive: bool = True) -> int:
    """Number of SAC updates this step, scaled by model-buffer fullness."""
    if not adaptive:
        return g_max
    frac = min(1.0, model_buffer_size / max(1, model_buffer_capacity))
    return max(1, int(round(g_max * frac)))


def build_macura(obs_dim: int, act_dim: int, cfg: dict, device: str = "cuda"):
    """Convenience builder: returns the shared SAC + ensemble for MACURA.

    The rollout logic above is stateless; training/train.py wires it together
    with `kappa_state = {}` carried across rounds.
    """
    from algorithms.sac import build_sac

    agent = build_sac(obs_dim, act_dim, cfg["sac"], device)
    dynamics_model = ens.build_ensemble(cfg["ensemble"], obs_dim, act_dim, device)
    return {"agent": agent, "dynamics_model": dynamics_model, "kappa_state": {}}
