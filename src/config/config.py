import os

# Single source of truth for every quantity. Nothing is hard-coded in the other
# .py files; each receives the relevant sub-dict (ENV, ENSEMBLE, SAC, ROLLOUT...).
# The task is Pogo (a planar one-legged gymnast) learning a BACKFLIP - an
# aggressive, unstable maneuver where a fixed-horizon rollout over-imagines the
# flight, so MACURA's uncertainty-adaptive truncation has a real job.

# EXPERIMENT
SEED                = 0
SEEDS               = [0, 1, 2]          # raise to 5 for final figures
ALGORITHMS          = ["macura", "mbpo", "m2ac", "sac"]
TOTAL_ENV_STEPS     = 20000              # a backflip is hard; raise for the full run
WARMUP_RANDOM_STEPS = 1000
EVAL_EVERY_STEPS    = 2000
EVAL_EPISODES       = 5
EVAL_SEEDS          = [100, 101, 102, 103, 104]   # SAME across all algorithms (fairness)

# DRIVE  -  bootstrap.setup() makes these folders; only small things go here.
DRIVE_ROOT = os.environ.get("MACURA_DRIVE_ROOT", "/content/drive/MyDrive/macura-backflip")

# ENV (Pogo backflip)
# The dense shaped reward is identical for real and imagined transitions.
# w_rotation is the CURRICULUM knob: 0 = just jump & balance, large = full flip.
REWARD = {
    "alive_bonus": 1.0,     # keeps reward positive
    "w_height":    4.0,     # reward foot clearance (get airborne so it CAN flip)
    "w_rotation":  1.0,     # reward spinning in flip_dir WHILE airborne (dense flip signal)
    "w_upright":   2.0,     # reward being upright when NOT airborne (land and hold)
    "w_action":    0.001,   # control-effort penalty
    "flip_dir":    1.0,     # +1 = one rotation direction (sign arbitrary, kept consistent)
    "foot_air":    0.20,    # foot clearance (m) at which the "airborne" gate saturates
    "height_cap":  0.5,     # cap on the foot-clearance jump reward
}
ENV = {
    "mjcf_scene":        "",     # "" -> bundled src/envs/assets/pogo.xml
    "action_repeat":     5,      # hold each action for 5 physics steps (0.01 s control)
    "max_episode_steps": 200,    # ~2 s episodes (200 * 5 * 0.002 s)
    "fail_torso_height": 0.10,   # torso center below this = collapsed / face-plant flat
    "init_height":       0.78,   # near-standing torso height at reset
    "init_noise":        0.04,   # small random pose + velocity perturbation at reset
    "reward":            REWARD,
}

# ENSEMBLE (probabilistic dynamics model, shared by all model-based algos)
ENSEMBLE = {
    "num_members": 5, "hidden_size": 200, "num_layers": 4, "activation": "silu",
    "learning_rate": 1.0e-3, "weight_decay": 1.0e-5, "batch_size": 256,
    "train_epochs_per_round": 8,
    "logvar_bounds": [-10.0, 0.5],   # bound logvars -> stable uGJS (Eq. 15-19)
    "deterministic": False, "propagation": "random_member",
}

# SAC backbone (IDENTICAL for all four algorithms - fairness)
SAC = {
    "gamma": 0.99, "tau": 0.005, "alpha": "auto",
    "actor_lr": 3.0e-4, "critic_lr": 3.0e-4,   # critic_lr ignored (SB3 uses one learning_rate)
    "hidden_size": 256, "batch_size": 256, "target_update_interval": 1,
    "gradient_steps_max": 8,        # Gmax in Eq. 22 (model-based UTD ceiling)
    "baseline_gradient_steps": 1,   # model-free SAC baseline UTD (~1 = standard SAC)
    "real_ratio": 0.05,             # fixed batch-level real mixing (Janner/MBPO; MACURA inherits)
}

# checkpoint selection & final eval (best = highest periodic greedy-eval return)
SELECTION = {"start_step": 4000, "eval_every": 2000, "final_eval_episodes": 10}

# ROLLOUT strategies (the ONLY thing that differs across algorithms)
ROLLOUT = {
    "freq_steps": 500, "num_rollouts": 200, "model_buffer_capacity": 200000,
    # MACURA: uncertainty-adaptive truncation (Algorithm 2)
    "macura": {"t_max": 10, "zeta": 0.95, "xi": 5.0, "adaptive_gradient_steps": True},
    # MBPO: fixed truncated-linear schedule (max_len matches MACURA t_max for fairness)
    "mbpo": {"rollout_schedule": [1, 10, 1000, 12000],
             "adaptive_gradient_steps": False, "fixed_gradient_steps": 8},
    # M2AC: fixed length + mask least-trustworthy transitions
    "m2ac": {"t_max": 10, "mask_fraction": 0.5, "uncertainty_penalty": 1.0,
             "fixed_gradient_steps": 8},
    "sac": {},
}

# EXPLORATION (kept CONSISTENT across algos to avoid the exploration confound)
EXPLORATION = {"type": "pink_noise", "scale": 0.3}

# assembled config the training functions consume
CFG = {
    "experiment": {
        "name": "macura_pogo_backflip", "seeds": SEEDS, "algorithms": ALGORITHMS,
        "total_env_steps": TOTAL_ENV_STEPS, "warmup_random_steps": WARMUP_RANDOM_STEPS,
        "eval_every_steps": EVAL_EVERY_STEPS, "eval_episodes": EVAL_EPISODES,
        "eval_seeds": EVAL_SEEDS, "drive_root": DRIVE_ROOT,
    },
    "env": ENV, "ensemble": ENSEMBLE, "sac": SAC,
    "selection": SELECTION, "rollout": ROLLOUT, "exploration": EXPLORATION,
}
