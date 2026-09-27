"""Experiment configuration: the single source of truth.

Every module receives one sub-dict of CFG (experiment, env, ensemble, sac, rollout, exploration, selection).
Pick the task and the run size with environment variables BEFORE this module is first imported:

    MACURA_TASK         race2 (the current study) | race3 (race2 + deadlier chute) | race | delivery2 | delivery | "" (cage)
    MACURA_STEPS        real environment steps per run (race tasks default to 50000)
    MACURA_SEEDS        e.g. "1" or "0 1 2"
    MACURA_ALGOS        e.g. "macura mbpo" (default: all four)
    MACURA_XI, MBPO_HORIZON, MBPO_UTD, M2AC_UTD, MACURA_EXPLORATION=equal     tuning / control runs
    MACURA_OUTPUT_ROOT  where checkpoints, logs, plots and videos go (default /kaggle/working/runs)
    MACURA_DEADLINE     unix time: a run still training then stops at its next evaluation and is saved

Why each task setting has its value (with the measurements behind it): docs/EXPERIMENTS.md.
"""

import os


def _env_list(name, default, cast):
    raw = os.environ.get(name, "")
    return [cast(x) for x in raw.replace(",", " ").split()] if raw.strip() else default


TASK = os.environ.get("MACURA_TASK", "").strip().lower()
RACE_TASKS = ("race", "race2", "race3")
RACE2_PROTOCOL_TASKS = ("race2", "race3")             # race3 runs the exact race2 protocol
PAPER_PROTOCOL_TASKS = ("delivery2", "race", "race2", "race3")

# ── experiment ────────────────────────────────────────────────────────────────────────────────────────
SEED = 0
SEEDS = _env_list("MACURA_SEEDS", [0], int)
ALGORITHMS = _env_list("MACURA_ALGOS", ["macura", "mbpo", "m2ac", "sac"], lambda x: x.strip().lower())
TOTAL_ENV_STEPS = int(os.environ.get("MACURA_STEPS", "") or (50000 if TASK in RACE_TASKS else 40000))
WARMUP_RANDOM_STEPS = 5000 if TASK in RACE2_PROTOCOL_TASKS else 500   # random actions before the first update (all algorithms)
EVAL_EVERY_STEPS = 1000
EVAL_EPISODES = 20                                      # fixed selection scenarios: seeds 100..119
EVAL_SEEDS = [100, 101, 102, 103, 104]                  # evaluate() uses base = EVAL_SEEDS[0]
OUTPUT_ROOT = os.environ.get("MACURA_OUTPUT_ROOT", "/kaggle/working/runs")
DEADLINE = float(os.environ.get("MACURA_DEADLINE", "") or 0) or None   # Kaggle kills a session at 12 h
RENDER = os.environ.get("MACURA_RENDER", "0") == "1"

# best checkpoint = highest periodic eval return (after start_step); its final test runs on FRESH
# scenarios (seeds 1000..1029), never on the selection scenarios it was picked on
SELECTION = {"start_step": 1000, "eval_every": 1000, "final_eval_episodes": 30, "final_eval_seed_base": 1000}

# ── environment: the base task is CAGE (land in an open-top cage); presets below switch task ──────────
REWARD = {
    "w_pos": 2.0, "pos_scale": 1.0,                     # be at the pad
    "w_level": 0.5, "w_spin": 0.01, "w_vel": 0.20, "w_ctrl": 0.01,
    "w_obs": 0.4, "obs_scale": 0.5,                     # obstacle proximity penalty
    "w_settle": 4.0, "settle_dist": 0.30, "settle_speed": 0.40,   # bonus only when close, slow and upright
    "w_cage": 0.8, "cage_scale": 0.15,                  # cage-wall proximity penalty (below its top)
}

ENV = {
    "mjcf_scene": "",                  # "" = the bundled src/envs/assets/drone.xml
    "action_repeat": 2,                # 50 Hz control
    "max_episode_steps": 400,
    "init_height": 2.0, "init_noise": 0.05, "init_tilt": 0.2, "init_spin": 0.3,
    "target_range_xy": 1.8, "pad_min_dist": 1.7, "pad_z": 0.2,
    "thrust_gain": 1.0,                # rotor command c -> thrust = empty-drone hover x (1 + c)
    "max_dist": 8.0, "fail_tilt": 0.0,
    # hidden process noise (real dynamics only)
    "wind_force": 0.4, "wind_correlation": 0.95, "actuator_noise": 0.04,
    "zone_radius": 1.0,                # "near the pad" radius of MACURA's near/far diagnostic
    # obstacles
    "n_obstacles": 0, "obstacle_radius": 0.3, "obstacle_min_clear": 0.5,
    "ring_radius": 0.85, "ring_gap_half_deg": 65.0,
    # landing envelope
    "land_radius": 0.5, "soft_speed": 0.8, "impact_height": 0.06, "hard_speed": 1.5,
    # cage
    "cage": True, "cage_radius": 0.8, "cage_height": 1.0, "cage_margin": 0.15, "cage_bars": 16,
    "start_height_min": 2.2, "start_height_max": 2.6, "start_offset": 1.0, "start_offset_min": 0.0,
    # delivery pieces (off here)
    "spawn_above_pad": False, "payload_max": 0.0,
    "vrs_loss": 0.0, "vrs_speed": 1.2, "vrs_full": 2.2, "vrs_escape": 1.0,
    "vrs_powered": False, "vrs_upright": 0.85, "vrs_thrust": 0.7,
    "scenery": False,
    "reward": REWARD,
}

DELIVERY = {
    "cage": False, "n_obstacles": 0, "spawn_above_pad": True,
    "start_height_min": 2.2, "start_height_max": 2.6, "start_offset": 0.5,
    "payload_max": 0.2,
    "vrs_loss": 0.35, "vrs_speed": 1.2, "vrs_full": 2.2, "vrs_escape": 1.0,
    "zone_radius": 1.0,
    "scenery": True,
}
DELIVERY2 = dict(DELIVERY, **{
    "vrs_powered": True, "vrs_upright": 0.85, "vrs_thrust": 0.7,
    "start_height_min": 3.0, "start_height_max": 4.0, "start_offset_min": 1.0, "start_offset": 2.0,
    "max_episode_steps": 500,
})
DELIVERY2_REWARD = {"w_level": 0.25}

# RACE: laps around a fixed 3-D course with a steep chute (src/envs/race_course.py)
RACE = {
    "race": True, "cage": False, "spawn_above_pad": False,
    "race_size": 3.0, "race_power": 4.0, "race_heights": (2.0, 4.0, 1.0, 1.5),
    "race_chute_u": 0.30, "race_chute_len": 0.25, "race_max_off": 1.5, "race_floor": 0.12,
    "n_obstacles": 2, "obstacle_radius": 0.30, "race_pillars": [(2.1, 2.1), (-2.1, -2.1)],
    "payload_max": 0.2,
    "vrs_loss": 0.35, "vrs_speed": 1.2, "vrs_full": 2.2, "vrs_escape": 1.0,
    "vrs_powered": True, "vrs_upright": 0.85, "vrs_thrust": 0.7,
    "wind_mean_max": 0.5,              # visible steady wind (N)
    "max_episode_steps": 500,          # 10 s flights
    "scenery": True,
}
RACE_REWARD = {"w_prog": 1.0,          # speed along the course, only near it
               "w_track": 1.0,         # minus metres outside the course tube
               "race_tube": 0.35, "race_tube_soft": 0.35,
               "w_level": 0.1,
               "w_crash": 500.0}
# RACE2 = RACE + altitude-hold stabiliser, racing-line sensor, 2.5 m off-course limit
RACE2 = dict(RACE, **{"ctrl_mode": "althold", "att_max_tilt_deg": 40.0, "alt_vz_max": 3.0, "alt_kz": 3.0,
                      "race_max_off": 2.5, "race_obs_course": True, "race_obs_ahead": 0.5})
# RACE3 = RACE2 with a deadlier chute: up to 50% lift lost (was 35%); where it starts (1.2 m/s) is unchanged.
# At 35% a moderate dive paid more than flying the chute safely; at 50% it clearly loses (docs/EXPERIMENTS.md).
RACE3 = dict(RACE2, vrs_loss=0.50)

if TASK == "delivery":
    ENV.update(DELIVERY)
elif TASK == "delivery2":
    ENV.update(DELIVERY2)
    ENV["reward"] = dict(REWARD, **DELIVERY2_REWARD)
elif TASK in RACE_TASKS:
    ENV.update({"race": RACE, "race2": RACE2, "race3": RACE3}[TASK])
    ENV["reward"] = dict(REWARD, **RACE_REWARD)
ENV["task"] = TASK

# ── learning: shared by all four algorithms ───────────────────────────────────────────────────────────
ENSEMBLE = {                           # probabilistic ensemble world model (MACURA / MBPO / M2AC)
    "num_members": 5, "hidden_size": 200, "num_layers": 4, "activation": "silu",
    "learning_rate": 1.0e-3, "weight_decay": 1.0e-5, "batch_size": 256,
    "train_epochs_per_round": 8,
    "logvar_bounds": [-10.0, 0.5],
}

SAC = {                                # the same SAC learner for every algorithm
    "gamma": 0.99, "tau": 0.005, "alpha": "auto", "actor_lr": 3.0e-4,
    "hidden_size": 256, "batch_size": 256, "target_update_interval": 1,
    "gradient_steps_max": 8,           # model-based updates per real step (MACURA: its Gmax, Eq. 22)
    "baseline_gradient_steps": 1,      # model-free SAC
    "real_ratio": 0.1,                 # share of real transitions in each model-based batch
}

ROLLOUT = {                            # how each method uses imagined data - the only difference between them
    "freq_steps": 500, "num_rollouts": 200, "model_buffer_capacity": 12000,
    "macura": {"t_max": 10, "zeta": 0.95, "xi": 0.4, "adaptive_gradient_steps": True},
    "mbpo": {"rollout_schedule": [1, 10, 500, 3000], "adaptive_gradient_steps": False, "fixed_gradient_steps": 8},
    "m2ac": {"t_max": 10, "mask_fraction": 0.5, "uncertainty_penalty": 0.1, "uncertainty": "ovr",
             "fixed_gradient_steps": 8},
    "sac": {},
}

EXPLORATION = {"type": "pink_noise", "scale": 0.3}

# ── paper protocol (MACURA paper App. D; M2AC as in Pan et al. 2020) ─────────────────────────────────
if TASK in PAPER_PROTOCOL_TASKS:
    ENSEMBLE.update({"num_members": 7, "num_elites": 5, "holdout_ratio": 0.2, "holdout_max": 5000,
                     "max_epochs": 10, "patience": 3})
    ROLLOUT.update({"freq_steps": 250, "num_rollouts": 25000, "model_buffer_capacity": 1_000_000})
    ROLLOUT["macura"] = dict(ROLLOUT["macura"], gradient_steps_max=16,
                             xi=float(os.environ.get("MACURA_XI", "") or (0.5 if TASK in RACE2_PROTOCOL_TASKS else 1.0)))
    _horizon = int(os.environ.get("MBPO_HORIZON", "") or 10)
    ROLLOUT["mbpo"] = dict(ROLLOUT["mbpo"], rollout_schedule=[1, _horizon, 500, 3000],
                           fixed_gradient_steps=int(os.environ.get("MBPO_UTD", "") or 8))
    ROLLOUT["m2ac"] = {"mode": "paper", "t_max": 10, "uncertainty_penalty": 1e-3, "uncertainty": "ovr",
                       "fixed_gradient_steps": int(os.environ.get("M2AC_UTD", "") or 8)}
    SAC["real_ratio"] = 0.05
    if os.environ.get("MACURA_EXPLORATION", "paper").strip().lower() != "equal":
        EXPLORATION["per_algo"] = {"macura": "pink_noise", "mbpo": "deterministic",
                                   "m2ac": "deterministic", "sac": "stochastic"}

if TASK in RACE2_PROTOCOL_TASKS:
    ROLLOUT["model_lifetime_rounds"] = 4   # imagined data expires after 4 model rounds (MACURA reference code)
    ROLLOUT["mbpo"]["rollout_schedule"] = [1, _horizon, WARMUP_RANDOM_STEPS, WARMUP_RANDOM_STEPS + 2500]

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
