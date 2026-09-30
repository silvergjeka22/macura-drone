"""SAC learner shared by all four algorithms (Stable-Baselines3, driven step by step from train.py)."""

from __future__ import annotations

import numpy as np

try:
    import torch
    import gymnasium as gym
    from stable_baselines3 import SAC
    from stable_baselines3.common.buffers import ReplayBuffer
except ImportError:
    torch = None
    gym = None
    SAC = None
    ReplayBuffer = None


class _DummyEnv(gym.Env if gym is not None else object):
    def __init__(self, obs_dim, act_dim):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (act_dim,), np.float32)

    def reset(self, *, seed=None, options=None):
        return np.zeros(self.observation_space.shape, np.float32), {}

    def step(self, action):
        obs = np.zeros(self.observation_space.shape, np.float32)
        return obs, 0.0, False, False, {}


def build_sac(obs_dim: int, act_dim: int, cfg: dict, device: str = "cuda", seed: int = 0):
    """cfg = the `sac` sub-config."""
    if SAC is None:
        raise ImportError("stable-baselines3 is required for SAC")
    device = device if torch.cuda.is_available() else "cpu"
    ent_coef = "auto" if cfg["alpha"] == "auto" else float(cfg["alpha"])
    agent = SAC(
        "MlpPolicy",
        _DummyEnv(obs_dim, act_dim),
        learning_rate=cfg["actor_lr"],
        buffer_size=cfg.get("buffer_size", 1_000_000),
        batch_size=cfg["batch_size"],
        tau=cfg["tau"],
        gamma=cfg["gamma"],
        ent_coef=ent_coef,
        target_update_interval=cfg["target_update_interval"],
        learning_starts=0,
        policy_kwargs={"net_arch": [cfg["hidden_size"], cfg["hidden_size"]]},
        device=device,
        seed=seed,
        verbose=0,
    )
    agent._setup_learn(total_timesteps=0, callback=None)
    return agent


def load_agent(path: str, obs_dim: int, act_dim: int, cfg: dict, device: str = "cpu"):
    """Load a saved policy; falls back to the weights only if the numpy version differs."""
    try:
        return SAC.load(path, device=device)
    except Exception:
        import io
        import zipfile
        agent = build_sac(obs_dim, act_dim, cfg, device)
        with zipfile.ZipFile(path) as z:
            state = torch.load(io.BytesIO(z.read("policy.pth")), map_location=agent.device)
        agent.policy.load_state_dict(state)
        return agent


def build_replay_buffer(agent, capacity: int):
    """A second SB3 buffer (the real data of the model-based agents)."""
    if ReplayBuffer is None:
        raise ImportError("stable-baselines3 is required for the replay buffer")
    return ReplayBuffer(
        int(capacity),
        agent.observation_space,
        agent.action_space,
        device=agent.device,
        n_envs=1,
        optimize_memory_usage=False,
        handle_timeout_termination=False,
    )


def add_to_buffer(buf, obs, act, next_obs, reward, done):
    buf.add(
        np.asarray(obs, np.float32).reshape(1, -1),
        np.asarray(next_obs, np.float32).reshape(1, -1),
        np.asarray(act, np.float32).reshape(1, -1),
        np.asarray([reward], np.float32),
        np.asarray([done], np.bool_),
        [{}],
    )


def add_transition(agent, obs, act, next_obs, reward, done):
    add_to_buffer(agent.replay_buffer, obs, act, next_obs, reward, done)


def sac_update(agent, num_updates: int, batch_size: int):
    if agent.replay_buffer.size() < batch_size or num_updates <= 0:
        return {"updates": 0}
    agent.train(gradient_steps=num_updates, batch_size=batch_size)
    return {"updates": num_updates}


class _MixedReplaySampler:
    """Replay buffer for model-based SAC: each batch is real_ratio real + the rest imagined."""

    def __init__(self, real_buf, model_buf, real_ratio, model_window=None):
        self.real_buf = real_buf
        self.model_buf = model_buf
        self.real_ratio = float(real_ratio)
        self.model_window = model_window

    def _sample_model(self, n, env=None):
        w = self.model_window
        if not w:
            return self.model_buf.sample(n, env=env)
        cap = self.model_buf.buffer_size
        idx = (self.model_buf.pos - 1 - np.random.randint(0, min(int(w), cap), size=n)) % cap
        return self.model_buf._get_samples(idx, env=env)

    def sample(self, batch_size, env=None):
        from stable_baselines3.common.type_aliases import ReplayBufferSamples
        n_real = int(round(self.real_ratio * batch_size))
        if self.real_buf.size() == 0:
            n_real = 0
        if self.model_buf.size() == 0:
            n_real = batch_size
        n_model = batch_size - n_real
        parts = []
        if n_real > 0:
            parts.append(self.real_buf.sample(n_real, env=env))
        if n_model > 0:
            parts.append(self._sample_model(n_model, env=env))
        if len(parts) == 1:
            return parts[0]

        # newer SB3 versions add optional fields that are None: pass those through
        def _merge(field):
            vals = [getattr(p, field) for p in parts]
            if all(torch.is_tensor(v) for v in vals):
                return torch.cat(vals, dim=0)
            return vals[0]
        return ReplayBufferSamples(*(_merge(f) for f in ReplayBufferSamples._fields))

    def __getattr__(self, name):
        if name in ("real_buf", "model_buf", "real_ratio", "model_window"):
            raise AttributeError(name)
        return getattr(self.model_buf, name)


def sac_update_mixed(agent, real_buf, model_buf, num_updates: int, batch_size: int,
                     real_ratio: float, rng=None, model_window=None) -> dict:
    """SB3's own agent.train() on mixed real / imagined batches (`rng` is unused)."""
    n_real = int(round(float(real_ratio) * batch_size))
    if num_updates <= 0:
        return {"updates": 0, "n_real_per_batch": n_real,
                "n_model_per_batch": batch_size - n_real,
                "real_pct": 100.0 * float(real_ratio),
                "imagined_pct": 100.0 * (1.0 - float(real_ratio))}
    saved = agent.replay_buffer
    agent.replay_buffer = _MixedReplaySampler(real_buf, model_buf, real_ratio, model_window)
    try:
        agent.train(gradient_steps=int(num_updates), batch_size=batch_size)
    finally:
        agent.replay_buffer = saved
    return {
        "updates": int(num_updates),
        "n_real_per_batch": n_real,
        "n_model_per_batch": batch_size - n_real,
        "real_pct":     100.0 * float(real_ratio),
        "imagined_pct": 100.0 * (1.0 - float(real_ratio)),
    }


def select_action(agent, obs: np.ndarray, evaluate: bool = False) -> np.ndarray:
    action, _ = agent.predict(np.asarray(obs, np.float32), deterministic=evaluate)
    return action


def select_actions(agent, obs_batch: np.ndarray, evaluate: bool = False) -> np.ndarray:
    actions, _ = agent.predict(np.asarray(obs_batch, np.float32), deterministic=evaluate)
    return actions
