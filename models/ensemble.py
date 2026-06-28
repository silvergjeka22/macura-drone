"""Probabilistic ensemble (PE) dynamics model — self-contained PyTorch.

A small ensemble of Gaussian MLPs predicting the next-state DELTA distribution,
shared by every model-based algorithm (MACURA, MBPO, M2AC). This replaces the
mbrl-lib dependency (unmaintained, not installable on Colab's Python 3.12) with
a compact, modern implementation. The math the algorithms rely on — per-member
predictive Gaussians for the GJS uncertainty — is exposed by `member_gaussians`.

Pure-function library: builders + helpers only.

Public functions:
    build_ensemble(cfg, obs_dim, act_dim, device) -> ens
    train_ensemble(ens, data, cfg)                 -> metrics
    predict(ens, obs, act)                         -> next_obs
    member_gaussians(ens, obs, act)                -> (means, variances)

`ens` is a dict: {model, optimizer, normalizer, device}. `data` is a dict of
numpy arrays {obs, act, next_obs}.

Note on the uncertainty: we predict the next-state DELTA (next_obs - obs). The
GJS divergence between members is invariant to the shared +obs shift, so
computing it on deltas equals computing it on next-states.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


# ── network ───────────────────────────────────────────────────────────────────
def _make_mlp(in_dim, out_dim, hidden, num_layers, activation):
    act = {"silu": nn.SiLU, "relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}.get(
        activation.lower(), nn.SiLU
    )
    layers, d = [], in_dim
    for _ in range(num_layers - 1):
        layers += [nn.Linear(d, hidden), act()]
        d = hidden
    layers += [nn.Linear(d, 2 * out_dim)]  # mean and logvar heads
    return nn.Sequential(*layers)


_Base = nn.Module if nn is not None else object


class GaussianEnsemble(_Base):
    """Ensemble of independent Gaussian MLPs (mean + bounded log-variance)."""

    def __init__(self, in_dim, out_dim, cfg):
        super().__init__()
        self.out_dim = out_dim
        self.members = nn.ModuleList(
            [
                _make_mlp(in_dim, out_dim, cfg["hidden_size"],
                          cfg["num_layers"], cfg["activation"])
                for _ in range(cfg["num_members"])
            ]
        )
        lo, hi = cfg["logvar_bounds"]
        self.min_logvar = nn.Parameter(torch.full((out_dim,), float(lo)), requires_grad=False)
        self.max_logvar = nn.Parameter(torch.full((out_dim,), float(hi)), requires_grad=False)

    def forward(self, x):
        """x: (B, in_dim). Returns means, logvars each (E, B, out_dim)."""
        means, logvars = [], []
        for m in self.members:
            out = m(x)
            mean, logvar = out[..., : self.out_dim], out[..., self.out_dim:]
            # soft-bound the log-variance for numerically stable uncertainty
            logvar = self.max_logvar - nn.functional.softplus(self.max_logvar - logvar)
            logvar = self.min_logvar + nn.functional.softplus(logvar - self.min_logvar)
            means.append(mean)
            logvars.append(logvar)
        return torch.stack(means, 0), torch.stack(logvars, 0)


# ── running input normalizer ──────────────────────────────────────────────────
class _Normalizer:
    def __init__(self, dim, device):
        self.mean = torch.zeros(dim, device=device)
        self.std = torch.ones(dim, device=device)

    def fit(self, x):
        self.mean = x.mean(0)
        self.std = x.std(0).clamp_min(1e-6)

    def __call__(self, x):
        return (x - self.mean) / self.std


# ── builders / helpers ────────────────────────────────────────────────────────
def build_ensemble(cfg: dict, obs_dim: int, act_dim: int, device: str = "cuda"):
    """`cfg` is the `ensemble:` sub-config. Returns the `ens` dict."""
    if torch is None:
        raise ImportError("torch is required for the ensemble")
    device = device if torch.cuda.is_available() else "cpu"
    model = GaussianEnsemble(obs_dim + act_dim, obs_dim, cfg).to(device)
    optim = torch.optim.Adam(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    return {
        "model": model,
        "optimizer": optim,
        "normalizer": _Normalizer(obs_dim + act_dim, device),
        "device": device,
        "cfg": cfg,
    }


def train_ensemble(ens: dict, data: dict, cfg: dict):
    """Train every member for `train_epochs_per_round` epochs on (s,a)->Δs.

    Each member sees a bootstrap resample of the data (epistemic disagreement).
    Returns {'nll': mean training NLL}.
    """
    model, optim, device = ens["model"], ens["optimizer"], ens["device"]
    obs = torch.as_tensor(data["obs"], dtype=torch.float32, device=device)
    act = torch.as_tensor(data["act"], dtype=torch.float32, device=device)
    nxt = torch.as_tensor(data["next_obs"], dtype=torch.float32, device=device)
    x = torch.cat([obs, act], dim=-1)
    y = nxt - obs                                   # predict the delta
    ens["normalizer"].fit(x)
    xn = ens["normalizer"](x)

    n, E = xn.shape[0], len(model.members)
    bs = min(cfg["batch_size"], n)
    last_nll = 0.0
    for _ in range(cfg["train_epochs_per_round"]):
        # per-member bootstrap indices
        boot = [torch.randint(0, n, (n,), device=device) for _ in range(E)]
        perm = torch.randperm(n, device=device)
        for start in range(0, n, bs):
            idx = perm[start:start + bs]
            optim.zero_grad()
            loss = 0.0
            for e in range(E):
                bi = boot[e][idx]
                mean_e, logvar_e = _forward_member(model, xn[bi], e)
                inv_var = torch.exp(-logvar_e)
                nll = 0.5 * ((mean_e - y[bi]) ** 2 * inv_var + logvar_e).sum(-1).mean()
                loss = loss + nll
            loss = loss / E
            loss.backward()
            optim.step()
            last_nll = float(loss.detach())
    return {"nll": last_nll}


def predict(ens: dict, obs: np.ndarray, act: np.ndarray):
    """Sample next_obs from a random member per row. Returns numpy (B, obs_dim)."""
    model, device = ens["model"], ens["device"]
    obs_t = torch.as_tensor(np.atleast_2d(obs), dtype=torch.float32, device=device)
    act_t = torch.as_tensor(np.atleast_2d(act), dtype=torch.float32, device=device)
    x = ens["normalizer"](torch.cat([obs_t, act_t], dim=-1))
    with torch.no_grad():
        means, logvars = model(x)                   # (E, B, D)
    E, B, _ = means.shape
    pick = torch.randint(0, E, (B,), device=device)
    mean = means[pick, torch.arange(B)]
    std = torch.exp(0.5 * logvars[pick, torch.arange(B)])
    delta = mean + std * torch.randn_like(std)
    next_obs = obs_t + delta
    return next_obs.cpu().numpy()


def member_gaussians(ens: dict, obs: np.ndarray, act: np.ndarray):
    """Per-member predictive Gaussians (means, variances), each (E, B, obs_dim).

    Returned in DELTA space — valid for GJS (shift-invariant). Feeds the GJS
    uncertainty in algorithms/macura.py.
    """
    model, device = ens["model"], ens["device"]
    obs_t = torch.as_tensor(np.atleast_2d(obs), dtype=torch.float32, device=device)
    act_t = torch.as_tensor(np.atleast_2d(act), dtype=torch.float32, device=device)
    x = ens["normalizer"](torch.cat([obs_t, act_t], dim=-1))
    with torch.no_grad():
        means, logvars = model(x)
    return means.cpu().numpy(), np.exp(logvars.cpu().numpy())


def save_ensemble(ens: dict, path: str):
    """Persist the ensemble for a best checkpoint: member weights + the input normalizer
    statistics (needed at predict time). No logic change — just serialization."""
    model, norm = ens["model"], ens["normalizer"]
    torch.save(
        {
            "model": model.state_dict(),
            "norm_mean": norm.mean.detach().cpu(),
            "norm_std": norm.std.detach().cpu(),
        },
        path,
    )
    return path


def load_ensemble(ens: dict, path: str):
    """Reload weights + normalizer stats saved by `save_ensemble` into an existing `ens` dict."""
    ckpt = torch.load(path, map_location=ens["device"])
    ens["model"].load_state_dict(ckpt["model"])
    ens["normalizer"].mean = ckpt["norm_mean"].to(ens["device"])
    ens["normalizer"].std = ckpt["norm_std"].to(ens["device"])
    return ens


def _forward_member(model, x, e):
    out = model.members[e](x)
    mean, logvar = out[..., : model.out_dim], out[..., model.out_dim:]
    logvar = model.max_logvar - nn.functional.softplus(model.max_logvar - logvar)
    logvar = model.min_logvar + nn.functional.softplus(logvar - model.min_logvar)
    return mean, logvar
