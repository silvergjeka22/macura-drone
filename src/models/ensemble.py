"""Probabilistic ensemble world model shared by MACURA, MBPO and M2AC."""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


def _make_mlp(in_dim, out_dim, hidden, num_layers, activation):
    act = {"silu": nn.SiLU, "relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}.get(
        activation.lower(), nn.SiLU
    )
    layers, d = [], in_dim
    for _ in range(num_layers - 1):
        layers += [nn.Linear(d, hidden), act()]
        d = hidden
    layers += [nn.Linear(d, 2 * out_dim)]
    return nn.Sequential(*layers)


_Base = nn.Module if nn is not None else object


class GaussianEnsemble(_Base):
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
        """x: (B, in_dim) -> means, logvars (E, B, out_dim)."""
        means, logvars = [], []
        for m in self.members:
            out = m(x)
            mean, logvar = out[..., : self.out_dim], out[..., self.out_dim:]
            logvar = self.max_logvar - nn.functional.softplus(self.max_logvar - logvar)
            logvar = self.min_logvar + nn.functional.softplus(logvar - self.min_logvar)
            means.append(mean)
            logvars.append(logvar)
        return torch.stack(means, 0), torch.stack(logvars, 0)


class _Normalizer:
    def __init__(self, dim, device):
        self.mean = torch.zeros(dim, device=device)
        self.std = torch.ones(dim, device=device)

    def fit(self, x):
        self.mean = x.mean(0)
        self.std = x.std(0).clamp_min(1e-6)

    def __call__(self, x):
        return (x - self.mean) / self.std


def build_ensemble(cfg: dict, obs_dim: int, act_dim: int, device: str = "cuda"):
    """cfg = the `ensemble` sub-config."""
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
        "target_normalizer": _Normalizer(obs_dim, device),
        "device": device,
        "cfg": cfg,
        "elites": None,
    }


def _train_epoch(model, optim, xn, yn, bs, device):
    """One epoch, every member on its own bootstrap resample. Returns the last batch loss."""
    n, E = xn.shape[0], len(model.members)
    last_nll = 0.0
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
            nll = 0.5 * ((mean_e - yn[bi]) ** 2 * inv_var + logvar_e).sum(-1).mean()
            loss = loss + nll
        loss = loss / E
        loss.backward()
        optim.step()
        last_nll = float(loss.detach())
    return last_nll


def _train_with_holdout(ens, xn, yn, cfg):
    """Train with a holdout set and early stopping; the best `num_elites` members become the elites."""
    import copy
    model, optim, device = ens["model"], ens["optimizer"], ens["device"]
    n, E = xn.shape[0], len(model.members)
    n_val = max(1, min(int(cfg.get("holdout_max", 5000)), int(n * float(cfg["holdout_ratio"]))))
    perm = torch.randperm(n, device=device)
    xv, yv, xt, yt = xn[perm[:n_val]], yn[perm[:n_val]], xn[perm[n_val:]], yn[perm[n_val:]]
    bs = min(cfg["batch_size"], xt.shape[0])

    def val_mse():
        with torch.no_grad():
            return torch.stack([((_forward_member(model, xv, e)[0] - yv) ** 2).mean() for e in range(E)])

    best = val_mse()
    best_state = [copy.deepcopy(m.state_dict()) for m in model.members]
    stale, epochs, last_nll = 0, 0, 0.0
    for _ in range(int(cfg.get("max_epochs", cfg["train_epochs_per_round"]))):
        last_nll = _train_epoch(model, optim, xt, yt, bs, device)
        epochs += 1
        cur = val_mse()
        improved = cur < best * 0.99
        for e in torch.nonzero(improved).flatten().tolist():
            best[e] = cur[e]
            best_state[e] = copy.deepcopy(model.members[e].state_dict())
        stale = 0 if bool(improved.any()) else stale + 1
        if stale >= int(cfg.get("patience", 3)):
            break
    for e, state in enumerate(best_state):
        model.members[e].load_state_dict(state)
    k = int(cfg.get("num_elites") or E)
    ens["elites"] = None if k >= E else torch.argsort(best)[:k]
    return {"nll": last_nll, "val_mse": float(best.mean()), "epochs": epochs}


def train_ensemble(ens: dict, data: dict, cfg: dict):
    """Fit (obs, act) -> next_obs - obs on all real data so far."""
    model, optim, device = ens["model"], ens["optimizer"], ens["device"]
    obs = torch.as_tensor(data["obs"], dtype=torch.float32, device=device)
    act = torch.as_tensor(data["act"], dtype=torch.float32, device=device)
    nxt = torch.as_tensor(data["next_obs"], dtype=torch.float32, device=device)
    x = torch.cat([obs, act], dim=-1)
    y = nxt - obs
    ens["normalizer"].fit(x)
    xn = ens["normalizer"](x)
    ens["target_normalizer"].fit(y)
    yn = ens["target_normalizer"](y)
    # static dims (payload, steady wind) are held fixed in rollouts and left out of the uncertainty
    dyn = y.std(0) > 0.0
    ens["dynamic"] = None if bool(dyn.all()) else dyn

    if float(cfg.get("holdout_ratio", 0.0)) > 0.0:
        return _train_with_holdout(ens, xn, yn, cfg)
    bs = min(cfg["batch_size"], xn.shape[0])
    last_nll = 0.0
    for _ in range(cfg["train_epochs_per_round"]):
        last_nll = _train_epoch(model, optim, xn, yn, bs, device)
    return {"nll": last_nll}


def predict(ens: dict, obs: np.ndarray, act: np.ndarray, return_pick: bool = False):
    """Sample next_obs from a random (elite) member per row; `return_pick` also returns that member."""
    model, device = ens["model"], ens["device"]
    obs_t = torch.as_tensor(np.atleast_2d(obs), dtype=torch.float32, device=device)
    act_t = torch.as_tensor(np.atleast_2d(act), dtype=torch.float32, device=device)
    x = ens["normalizer"](torch.cat([obs_t, act_t], dim=-1))
    with torch.no_grad():
        means, logvars = model(x)
        el = ens.get("elites")
        if el is not None:
            means, logvars = means[el], logvars[el]
    E, B, _ = means.shape
    pick = torch.randint(0, E, (B,), device=device)
    rows = torch.arange(B, device=device)
    mean = means[pick, rows]
    std = torch.exp(0.5 * logvars[pick, rows])
    delta_n = mean + std * torch.randn_like(std)
    tn = ens["target_normalizer"]
    next_obs = obs_t + (tn.mean + tn.std * delta_n)
    dyn = ens.get("dynamic")
    if dyn is not None:
        next_obs[:, ~dyn] = obs_t[:, ~dyn]
    if return_pick:
        return next_obs.cpu().numpy(), pick.cpu().numpy()
    return next_obs.cpu().numpy()


def member_gaussians(ens: dict, obs: np.ndarray, act: np.ndarray,
                     denormalize: bool = False):
    """Per-member (means, variances), each (E, B, D), in normalised delta space (denormalize=True: physical)."""
    model, device = ens["model"], ens["device"]
    obs_t = torch.as_tensor(np.atleast_2d(obs), dtype=torch.float32, device=device)
    act_t = torch.as_tensor(np.atleast_2d(act), dtype=torch.float32, device=device)
    x = ens["normalizer"](torch.cat([obs_t, act_t], dim=-1))
    with torch.no_grad():
        means, logvars = model(x)
        el = ens.get("elites")
        if el is not None:
            means, logvars = means[el], logvars[el]
        if denormalize:
            tn = ens["target_normalizer"]
            means = tn.mean + tn.std * means
            logvars = logvars + 2.0 * torch.log(tn.std)
        dyn = ens.get("dynamic")
        if dyn is not None:
            means, logvars = means[..., dyn], logvars[..., dyn]
    return means.cpu().numpy(), np.exp(logvars.cpu().numpy())


def save_ensemble(ens: dict, path: str):
    """Member weights + normaliser statistics + dynamic mask + elites."""
    model, norm, tnorm = ens["model"], ens["normalizer"], ens["target_normalizer"]
    torch.save(
        {
            "model": model.state_dict(),
            "norm_mean": norm.mean.detach().cpu(),
            "norm_std": norm.std.detach().cpu(),
            "tnorm_mean": tnorm.mean.detach().cpu(),
            "tnorm_std": tnorm.std.detach().cpu(),
            "dynamic": None if ens.get("dynamic") is None else ens["dynamic"].detach().cpu(),
            "elites": None if ens.get("elites") is None else ens["elites"].detach().cpu(),
        },
        path,
    )
    return path


def load_ensemble(ens: dict, path: str):
    ckpt = torch.load(path, map_location=ens["device"])
    ens["model"].load_state_dict(ckpt["model"])
    ens["normalizer"].mean = ckpt["norm_mean"].to(ens["device"])
    ens["normalizer"].std = ckpt["norm_std"].to(ens["device"])
    if "tnorm_mean" in ckpt:
        ens["target_normalizer"].mean = ckpt["tnorm_mean"].to(ens["device"])
        ens["target_normalizer"].std = ckpt["tnorm_std"].to(ens["device"])
    if ckpt.get("dynamic") is not None:
        ens["dynamic"] = ckpt["dynamic"].to(ens["device"])
    if ckpt.get("elites") is not None:
        ens["elites"] = ckpt["elites"].to(ens["device"])
    return ens


def _forward_member(model, x, e):
    out = model.members[e](x)
    mean, logvar = out[..., : model.out_dim], out[..., model.out_dim:]
    logvar = model.max_logvar - nn.functional.softplus(model.max_logvar - logvar)
    logvar = model.min_logvar + nn.functional.softplus(logvar - model.min_logvar)
    return mean, logvar
