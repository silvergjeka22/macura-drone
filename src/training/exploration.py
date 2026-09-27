"""Exploration while collecting real data: pink / white noise on the policy mean, deterministic,
or sampling the SAC policy (stochastic)."""

from __future__ import annotations

import numpy as np

from src.algorithms import sac as sac_mod


class _WhiteNoise:
    def __init__(self, act_dim, scale, seed=0):
        self.act_dim, self.scale = act_dim, float(scale)
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def sample(self):
        return self.scale * self.rng.normal(size=self.act_dim)


class _PinkNoise:
    """1/f noise per action dim, one sequence of `horizon` steps per episode (Eberhard et al. 2023)."""

    def __init__(self, act_dim, horizon, scale, seed=0):
        self.act_dim = act_dim
        self.horizon = max(int(horizon), 2)
        self.scale = float(scale)
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.seq = self._gen()

    def _gen(self):
        n = self.horizon
        f = np.fft.rfftfreq(n)
        f[0] = f[1]
        cols = []
        for _ in range(self.act_dim):
            spec = (self.rng.normal(size=f.shape) + 1j * self.rng.normal(size=f.shape)) / np.sqrt(f)
            x = np.fft.irfft(spec, n=n)
            cols.append(x / (x.std() + 1e-9))
        return np.stack(cols, axis=1)

    def reset(self):
        self.t = 0
        self.seq = self._gen()

    def sample(self):
        v = self.seq[self.t % self.horizon]
        self.t += 1
        return self.scale * v


class _NoNoise:
    def reset(self):
        pass

    def sample(self):
        return 0.0


class _Stochastic(_NoNoise):
    stochastic = True


def algo_exploration(expl_cfg, algo_name):
    """The scheme `algo_name` uses: `per_algo` (paper protocol) or the shared `type`."""
    t = (expl_cfg.get("per_algo") or {}).get(algo_name, expl_cfg.get("type", "white_noise"))
    return {**expl_cfg, "type": t}


def make_noise(expl_cfg, act_dim, horizon, seed):
    t = expl_cfg.get("type", "white_noise")
    s = float(expl_cfg.get("scale", 0.1))
    if t == "deterministic":
        return _NoNoise()
    if t == "stochastic":
        return _Stochastic()
    if t == "pink_noise":
        return _PinkNoise(act_dim, horizon, s, seed)
    return _WhiteNoise(act_dim, s, seed)


def explore(agent, obs, noise_proc):
    if getattr(noise_proc, "stochastic", False):
        return sac_mod.select_action(agent, obs, evaluate=False)
    act = sac_mod.select_action(agent, obs, evaluate=True)
    return np.clip(act + noise_proc.sample(), -1.0, 1.0)
