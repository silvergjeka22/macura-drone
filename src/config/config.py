"""All settings of the study (race3 task, MACURA paper protocol).

Environment variables change a run: MACURA_STEPS, MACURA_SEEDS, MACURA_ALGOS, MACURA_OUTPUT_ROOT, MACURA_DEADLINE,
MBPO_UTD, M2AC_UTD, MACURA_XI, MBPO_HORIZON, MACURA_EXPLORATION=equal.
"""

import os


def _env(name, default):
    return os.environ.get(name, "").strip() or default


def _env_list(name, default, cast):
    raw = os.environ.get(name, "")
    return [cast(x) for x in raw.replace(",", " ").split()] if raw.strip() else default


TASK = "race3"

# experiment
SEED = 0
SEEDS = _env_list("MACURA_SEEDS", [0], int)
ALGORITHMS = _env_list("MACURA_ALGOS", ["macura", "mbpo", "m2ac", "sac"], lambda x: x.strip().lower())
TOTAL_ENV_STEPS = int(_env("MACURA_STEPS", 50000))
WARMUP_RANDOM_STEPS = 5000             # random actions before the first update (all algorithms)
EVAL_EVERY_STEPS = 1000
EVAL_EPISODES = 20                     # selection scenarios: seeds 100..119
EVAL_SEEDS = [100, 101, 102, 103, 104]
OUTPUT_ROOT = _env("MACURA_OUTPUT_ROOT", "/kaggle/working/runs")
DEADLINE = float(_env("MACURA_DEADLINE", 0)) or None
# the best checkpoint (highest evaluation return) is tested on fresh scenarios 1000..1029
SELECTION = {"start_step": 1000, "eval_every": 1000, "final_eval_episodes": 30, "final_eval_seed_base": 1000}

# environment: laps around a 3-D course with a chute
REWARD = {
    "w_prog": 1.0,                     # speed along the course, only near it
    "w_track": 1.0,                    # minus metres outside the course tube
    "race_tube": 0.35, "race_tube_soft": 0.35,
    "w_level": 0.1, "w_spin": 0.01, "w_ctrl": 0.01,
    "w_crash": 500.0,
}

ENV = {
    "task": TASK,
    "action_repeat": 2,                # 50 Hz control
    "max_episode_steps": 500,          # 10 s flights
    "init_noise": 0.05, "init_tilt": 0.2, "init_spin": 0.3,
    "thrust_gain": 1.0,
    "max_dist": 8.0,
    # hidden process noise
    "wind_force": 0.4, "wind_correlation": 0.95, "actuator_noise": 0.04,
    "wind_mean_max": 0.5,              # steady wind, visible to the drone
    "payload_max": 0.2,                # package, visible to the drone
    # the course (src/envs/race_course.py)
    "race_size": 3.0, "race_power": 4.0, "race_heights": (2.0, 4.0, 1.0, 1.5),
    "race_chute_u": 0.30, "race_chute_len": 0.25,
    "race_max_off": 2.5, "race_floor": 0.12,
    "race_pillars": [(2.1, 2.1), (-2.1, -2.1)], "obstacle_radius": 0.30,
    "race_obs_ahead": 0.5,             # racing-line sensor looks 0.5 m ahead
    # lift loss when sinking fast, upright and pushing (the chute trap)
    "vrs_loss": 0.50, "vrs_speed": 1.2, "vrs_full": 2.2, "vrs_escape": 1.0, "vrs_upright": 0.85, "vrs_thrust": 0.7,
    # on-board altitude-hold stabiliser
    "att_max_tilt_deg": 40.0, "alt_vz_max": 3.0, "alt_kz": 3.0,
    "reward": REWARD,
}

# learning: the same for all four algorithms
ENSEMBLE = {
    "num_members": 7, "num_elites": 5, "hidden_size": 200, "num_layers": 4, "activation": "silu",
    "learning_rate": 1.0e-3, "weight_decay": 1.0e-5, "batch_size": 256, "train_epochs_per_round": 8,
    "logvar_bounds": [-10.0, 0.5],
    "holdout_ratio": 0.2, "holdout_max": 5000, "max_epochs": 10, "patience": 3,
}

SAC = {
    "gamma": 0.99, "tau": 0.005, "alpha": "auto", "actor_lr": 3.0e-4,
    "hidden_size": 256, "batch_size": 256, "target_update_interval": 1,
    "gradient_steps_max": 8,
    "baseline_gradient_steps": 1,      # model-free SAC
    "real_ratio": 0.05,                # share of real transitions in each model-based batch
}

_horizon = int(_env("MBPO_HORIZON", 10))
ROLLOUT = {                            # how each method uses imagined data - the only difference between them
    "freq_steps": 250, "num_rollouts": 25000, "model_buffer_capacity": 1_000_000,
    "model_lifetime_rounds": 4,        # imagined data expires after 4 model rounds
    "macura": {"t_max": 10, "zeta": 0.95, "xi": float(_env("MACURA_XI", 0.5)),
               "adaptive_gradient_steps": True, "gradient_steps_max": 16},
    "mbpo": {"rollout_schedule": [1, _horizon, WARMUP_RANDOM_STEPS, WARMUP_RANDOM_STEPS + 2500],
             "adaptive_gradient_steps": False, "fixed_gradient_steps": int(_env("MBPO_UTD", 8))},
    "m2ac": {"t_max": 10, "uncertainty_penalty": 1e-3, "fixed_gradient_steps": int(_env("M2AC_UTD", 8))},
    "sac": {},
}

EXPLORATION = {"type": "pink_noise", "scale": 0.3}
if _env("MACURA_EXPLORATION", "paper").lower() != "equal":
    EXPLORATION["per_algo"] = {"macura": "pink_noise", "mbpo": "deterministic", "m2ac": "deterministic",
                               "sac": "stochastic"}

CFG = {
    "experiment": {
        "name": "macura_drone", "seeds": SEEDS, "algorithms": ALGORITHMS,
        "total_env_steps": TOTAL_ENV_STEPS, "warmup_random_steps": WARMUP_RANDOM_STEPS,
        "eval_every_steps": EVAL_EVERY_STEPS, "eval_episodes": EVAL_EPISODES,
        "eval_seeds": EVAL_SEEDS, "output_root": OUTPUT_ROOT, "deadline": DEADLINE,
    },
    "env": ENV, "ensemble": ENSEMBLE, "sac": SAC,
    "selection": SELECTION, "rollout": ROLLOUT, "exploration": EXPLORATION,
}
