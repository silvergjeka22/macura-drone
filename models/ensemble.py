"""Probabilistic ensemble (PE) dynamics model — thin wrapper over mbrl-lib.

The PE is shared by every model-based algorithm (MACURA, MBPO, M2AC), so it
lives here once. We build on `mbrl.models.GaussianMLP` (the same model class the
paper uses via mbrl-lib) to guarantee algorithmic fairness and avoid reinventing
a probabilistic ensemble.

Pure-function library: builders + helpers, no orchestration.

Public functions:
    build_ensemble(cfg, obs_dim, act_dim, device) -> dynamics_model
    train_ensemble(dynamics_model, replay_buffer, cfg)      -> train_metrics
    predict(dynamics_model, obs, act)                        -> next_obs, info
    member_gaussians(dynamics_model, obs, act)              -> (means, variances)

The last helper exposes PER-MEMBER predictive Gaussians, which the MACURA /
M2AC uncertainty measures consume (see algorithms/macura.py).

NOTE (Colab binding): the exact mbrl-lib call signatures are pinned to the
installed version. The functions below follow mbrl-lib's documented API; if an
import/signature drifts, this single file is where to adjust it (see tasks.md).
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import mbrl.models as mbrl_models
    import mbrl.util.common as mbrl_common
except ImportError:  # allow inspection without the full stack installed
    torch = None
    mbrl_models = None
    mbrl_common = None


def build_ensemble(cfg: dict, obs_dim: int, act_dim: int, device: str = "cuda"):
    """Build a GaussianMLP probabilistic ensemble wrapped in a OneDTransition model.

    `cfg` is the `ensemble:` sub-config. Returns an mbrl `OneDTransitionRewardModel`
    when a reward head is desired; here we predict next-state deltas only and use
    the KNOWN analytic reward (see envs.drone_env.known_reward_fn) for fairness.
    """
    if mbrl_models is None:
        raise ImportError("mbrl-lib is required to build the ensemble")

    member = mbrl_models.GaussianMLP(
        in_size=obs_dim + act_dim,
        out_size=obs_dim,                     # predict next-state delta
        device=device,
        num_layers=cfg["num_layers"],
        ensemble_size=cfg["num_members"],
        hid_size=cfg["hidden_size"],
        activation_fn_cfg={"_target_": f"torch.nn.{_act_name(cfg['activation'])}"},
        deterministic=cfg["deterministic"],
        propagation_method=cfg["propagation"],
        learn_logvar_bounds=False,
    )
    # bound the predicted log-variances -> numerically stable uGJS (Eq. 15-19)
    lo, hi = cfg["logvar_bounds"]
    member.min_logvar.data.fill_(lo)
    member.max_logvar.data.fill_(hi)

    dynamics_model = mbrl_models.OneDTransitionRewardModel(
        member,
        target_is_delta=True,                 # learn s' - s, standard in MBPO
        normalize=True,                       # observation normalization (faster)
        learned_rewards=False,                # we use the known analytic reward
        obs_process_fn=None,
        no_delta_list=None,
    )
    return dynamics_model


def train_ensemble(dynamics_model, replay_buffer, cfg: dict):
    """Train the ensemble for a round on the environment replay buffer.

    Returns a dict of training metrics (train/val loss) for logging.
    """
    if mbrl_common is None:
        raise ImportError("mbrl-lib is required to train the ensemble")

    model_trainer = mbrl_models.ModelTrainer(
        dynamics_model,
        optim_lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    dataset_train, dataset_val = mbrl_common.get_basic_buffer_iterators(
        replay_buffer,
        batch_size=cfg["batch_size"],
        val_ratio=0.2,
    )
    train_losses, val_losses = model_trainer.train(
        dataset_train,
        dataset_val=dataset_val,
        num_epochs=cfg["train_epochs_per_round"],
    )
    return {
        "train_loss": float(np.mean(train_losses[-1])) if train_losses else None,
        "val_loss": float(np.mean(val_losses[-1])) if val_losses else None,
    }


def predict(dynamics_model, obs: np.ndarray, act: np.ndarray):
    """One-step model prediction. Returns (next_obs, info).

    Samples a single ensemble member per call (propagation set at build time),
    used for generating branched rollouts.
    """
    if torch is None:
        raise ImportError("torch/mbrl-lib required for prediction")
    model_state = dynamics_model.reset(
        torch.as_tensor(obs, dtype=torch.float32, device=_device(dynamics_model)),
        rng=torch.Generator(device=_device(dynamics_model)),
    )
    with torch.no_grad():
        next_obs, _, _, model_state = dynamics_model.sample(
            torch.as_tensor(act, dtype=torch.float32, device=_device(dynamics_model)),
            model_state,
            deterministic=False,
        )
    return next_obs.cpu().numpy(), {"model_state": model_state}


def member_gaussians(dynamics_model, obs: np.ndarray, act: np.ndarray):
    """Per-member predictive Gaussians at (obs, act).

    Returns (means, variances), each of shape (num_members, batch, obs_dim).
    These feed the GJS uncertainty in algorithms/macura.py. We query the
    underlying GaussianMLP directly to get ALL members' outputs (not a sampled
    one), then undo normalization so means live in observation space.
    """
    if torch is None:
        raise ImportError("torch/mbrl-lib required")
    device = _device(dynamics_model)
    x = np.concatenate([np.atleast_2d(obs), np.atleast_2d(act)], axis=-1)
    x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
    if dynamics_model.input_normalizer is not None:
        x_t = dynamics_model.input_normalizer.normalize(x_t)
    with torch.no_grad():
        # GaussianMLP.forward returns (mean, logvar) with leading ensemble dim
        mean, logvar = dynamics_model.model.forward(x_t, use_propagation=False)
    means = mean.cpu().numpy()
    variances = np.exp(logvar.cpu().numpy())
    return means, variances


# ── small internals ───────────────────────────────────────────────────────────
def _act_name(name: str) -> str:
    return {"silu": "SiLU", "relu": "ReLU", "tanh": "Tanh", "elu": "ELU"}.get(
        name.lower(), "SiLU"
    )


def _device(dynamics_model) -> str:
    try:
        return str(next(dynamics_model.model.parameters()).device)
    except Exception:
        return "cpu"
