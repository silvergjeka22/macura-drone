import os

# Single source of truth for every quantity. Nothing is hard-coded in the other .py
# files; each receives the relevant sub-dict (ENV, ENSEMBLE, SAC, ROLLOUT...).
#
# TASK: a quadrotor learns to RECOVER from a tumbling start, fly to a target, and hold a
# stable hover. The high-angular-rate recovery + fast approach are aggressive - exactly
# where a learned model is uncertain, so a fixed-horizon rollout (MBPO) over-imagines and
# MACURA's uncertainty-adaptive truncation has a real job. The four algorithms
# (MACURA, MBPO, M2AC, SAC) share the SAC backbone, the ensemble, pink-noise exploration
# and eval seeds - ONLY the rollout strategy differs.

# EXPERIMENT
SEED                = 0
SEEDS               = [0, 1, 2, 3, 4, 5, 6, 7]   # 8 seeds - beat the noise in a short run (fair: same for all)
ALGORITHMS          = ["macura", "mbpo", "m2ac", "sac"]
TOTAL_ENV_STEPS     = 28000              # SHORT-but-fair budget: small enough to be cheap, large enough for
                                         # the model-based-vs-SAC win + MACURA's stability edge to show, and
                                         # for MACURA to reach full adaptive UTD (~step 8k). [0]+~2000 = smoke.
WARMUP_RANDOM_STEPS = 500
EVAL_EVERY_STEPS    = 1000               # ~28 eval points over the run
EVAL_EPISODES       = 5                  # 5 (not 2) -> lower-variance eval so a small MACURA gap is legible
EVAL_SEEDS          = [100, 101, 102, 103, 104]   # SAME across all algorithms (fairness)

# OUTPUT ROOT  -  bootstrap.setup() makes these folders (checkpoints/logs/plots/videos).
# /kaggle/working is the only writable dir Kaggle saves as the kernel's downloadable output,
# so results go to /kaggle/working/runs. Override with MACURA_OUTPUT_ROOT if you like.
OUTPUT_ROOT = os.environ.get("MACURA_OUTPUT_ROOT", "/kaggle/working/runs")

# RENDERING - OFF in the training kernel (MUJOCO_GL=disable; loading libOSMesa next to
# torch/SB3 segfaults). The env-preview video renders in an isolated osmesa subprocess.
RENDER = os.environ.get("MACURA_RENDER", "0") == "1"

# ENV (drone recover-and-reach). Dense reward, analytic in (obs, action) - identical for
# real and imagined transitions. The drone spawns TILTED and TUMBLING (init_tilt/init_spin)
# and must recover, then reach the target and hold hover.
REWARD = {
    "w_pos":     2.0,    # reward being AT the target (max, via exp(-dist/scale))
    "pos_scale": 1.0,    # distance scale (m) of the proximity reward
    "w_level":   0.5,    # reward staying level (body-z world component) -> drives recovery
    "w_spin":    0.01,   # penalize angular velocity
    "w_vel":     0.05,   # penalize speed (so it STOPS at the target, not overshoot)
    "w_ctrl":    0.01,   # mild control penalty (thrust deviation from hover)
}
ENV = {
    "mjcf_scene":        "",     # "" -> bundled src/envs/assets/drone.xml
    "action_repeat":     2,      # 50 Hz control (timestep 0.01 * 2)
    "max_episode_steps": 250,    # ~5 s episodes
    "init_height":       1.5,    # spawn height (m) - room to recover before the ground
    "init_noise":        0.05,   # small random pose + velocity perturbation at reset
    "init_tilt":         0.9,    # aggressive: random start tilt up to ~0.9 rad (~52 deg)
    "init_spin":         2.5,    # aggressive: random start angular velocity up to 2.5 rad/s
    "target_range_xy":   2.0,    # target sampled in x,y in [-2, 2] m
    "target_z_min":      0.8,
    "target_z_max":      2.0,
    "thrust_gain":       1.0,    # action*gain about hover: action 0 = hover, +-1 = 0..2x hover
    "max_dist":          8.0,    # flew away this far = crashed
    "fail_tilt":        -1.0,    # tilt-termination DISABLED: recovery = being tilted then leveling
    "fail_height":       0.15,   # hit the ground = crashed
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
    "real_ratio": 0.05,             # fixed within-batch real mixing (Janner/MBPO; MACURA inherits)
}

# checkpoint selection & final eval (best = highest periodic greedy-eval return)
SELECTION = {"start_step": 1000, "eval_every": 1000, "final_eval_episodes": 10}

# ROLLOUT strategies (the ONLY thing that differs across algorithms).
# model_buffer_capacity is chosen so MACURA fills it (and thus reaches full adaptive UTD,
# Eq. 22) within roughly the first third of the run - NOT scaled linearly with
# TOTAL_ENV_STEPS: a too-large buffer would leave MACURA perpetually below MBPO/M2AC's fixed
# update budget. At 28k steps (capacity 14k) MACURA reaches |D_mod|_max around step ~8k.
ROLLOUT = {
    "freq_steps": 500, "num_rollouts": 200, "model_buffer_capacity": 14000,
    # MACURA: uncertainty-adaptive truncation (Algorithm 2)
    "macura": {"t_max": 10, "zeta": 0.95, "xi": 5.0, "adaptive_gradient_steps": True},
    # MBPO: fixed truncated-linear schedule; ramp scaled to REACH horizon 10 within the run.
    "mbpo": {"rollout_schedule": [1, 10, 500, 3000],
             "adaptive_gradient_steps": False, "fixed_gradient_steps": 8},
    # M2AC: fixed length + mask least-trustworthy transitions. "ovr" = M2AC's own one-vs-rest
    # disagreement (paper-faithful); "gjs" = reuse MACURA's GJS (same-signal ablation).
    "m2ac": {"t_max": 10, "mask_fraction": 0.5, "uncertainty_penalty": 1.0,
             "uncertainty": "ovr", "fixed_gradient_steps": 8},
    "sac": {},
}

# EXPLORATION (kept CONSISTENT across algos to avoid the exploration confound)
EXPLORATION = {"type": "pink_noise", "scale": 0.3}

# assembled config the training functions consume
CFG = {
    "experiment": {
        "name": "macura_drone", "seeds": SEEDS, "algorithms": ALGORITHMS,
        "total_env_steps": TOTAL_ENV_STEPS, "warmup_random_steps": WARMUP_RANDOM_STEPS,
        "eval_every_steps": EVAL_EVERY_STEPS, "eval_episodes": EVAL_EPISODES,
        "eval_seeds": EVAL_SEEDS, "output_root": OUTPUT_ROOT,
    },
    "env": ENV, "ensemble": ENSEMBLE, "sac": SAC,
    "selection": SELECTION, "rollout": ROLLOUT, "exploration": EXPLORATION,
}
