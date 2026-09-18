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
# SEEDS and ALGORITHMS are overridable from the environment (set them in a notebook cell BEFORE
# `from src.bootstrap import setup`, i.e. before config is first imported). Handy for a fast smoke
# test without editing code: e.g. `os.environ["MACURA_SEEDS"] = "0"`.
#   default (no env):  4 algos x 5 seeds x 25k steps ~= 9-10h  -> fits ONE Kaggle 12h commit,
#                      all plots + videos in a single run (a commit that runs past 12h is killed
#                      and saves NOTHING, so do not enlarge this without splitting across commits).
def _env_list(name, default, cast):
    raw = os.environ.get(name, "")
    if not raw.strip():
        return default
    return [cast(x) for x in raw.replace(",", " ").split()]

SEED                = 0
SEEDS               = _env_list("MACURA_SEEDS", [0, 1, 2, 3, 4], int)   # 5-seed study: enough for IQM +
                                         # bootstrap CIs, the credible basis for a "MACURA wins" claim
                                         # (single-seed spot-check: MACURA_SEEDS="0").
ALGORITHMS          = _env_list("MACURA_ALGOS", ["macura", "mbpo", "m2ac", "sac"],
                                lambda x: x.strip().lower())
TOTAL_ENV_STEPS     = 25000              # 4 algos x 5 seeds x 25k with UTD 4 ~= 5-6h -> fits one 12h Kaggle commit,
                                         # every plot + the IQM/crash-CI comparison in a single run.
WARMUP_RANDOM_STEPS = 500
EVAL_EVERY_STEPS    = 1000               # ~20 eval points over the run
EVAL_EPISODES       = 20                 # 20 FIXED-seed episodes/eval -> per-point crash noise ~sqrt(p(1-p)/20)
                                         # ~=0.11 (was ~0.22 at 5): the learning curve reflects the POLICY, not
                                         # scenario luck. Cheap (eval is a small fraction of runtime); the UTD cut
                                         # below more than pays for it.
EVAL_SEEDS          = [100, 101, 102, 103, 104]   # base for the fixed eval scenarios (evaluate() uses
                                         # base+i, i.e. seeds 100..119 for 20 episodes) - SAME across all algorithms.

# OUTPUT ROOT  -  bootstrap.setup() makes these folders (checkpoints/logs/plots/videos).
# /kaggle/working is the only writable dir Kaggle saves as the kernel's downloadable output,
# so results go to /kaggle/working/runs. Override with MACURA_OUTPUT_ROOT if you like.
OUTPUT_ROOT = os.environ.get("MACURA_OUTPUT_ROOT", "/kaggle/working/runs")

# RENDERING - OFF in the training kernel (MUJOCO_GL=disable; loading libOSMesa next to
# torch/SB3 segfaults). The env-preview video renders in an isolated osmesa subprocess.
RENDER = os.environ.get("MACURA_RENDER", "0") == "1"

# ENV (drone in wind: fly through obstacles, land softly on a pad). Dense reward, analytic in
# (obs, action) - identical for real and imagined transitions. PROCESS NOISE (gusts + actuator
# noise) makes the learned model uncertain across the whole trajectory: a fixed-horizon rollout
# (MBPO) over-imagines and diverges, while MACURA truncates where the ensemble disagrees. This
# is MACURA's strongest honest case on the drone (paper App. D.4: MACURA excels under process noise).
REWARD = {
    "w_pos":     2.0,    # reward being AT the landing pad (max, via exp(-dist/scale))
    "pos_scale": 1.0,    # distance scale (m) of the proximity reward (was 1.5 -> steeper pull INTO the pad)
    "w_level":   0.5,    # reward staying upright (body-z world component)
    "w_spin":    0.01,   # penalize angular velocity
    "w_vel":     0.20,   # penalize speed (was 0.10 -> actually rewards slowing down to land)
    "w_ctrl":    0.01,   # mild control penalty (thrust deviation from hover)
    "w_obs":     0.4,    # obstacle-avoidance penalty (softened so it doesn't destabilize learning)
    "obs_scale": 0.5,    # distance scale (m) of the obstacle penalty
    # SETTLE bonus: a reward "well" that fires ONLY inside the landing envelope (close AND slow AND
    # upright) so the policy is actually driven to REACH the pad, not just hover ~1 m away. Pure
    # function of obs (dist, speed, up_z) -> real == imagined kept, and applied to all 4 algos (fair).
    "w_settle":     4.0,   # strength of the landing-envelope bonus
    "settle_dist":  0.30,  # distance scale (m): tight, so it only rewards being ON the pad
    "settle_speed": 0.40,  # speed scale (m/s): only rewards a slow, controlled arrival
}
ENV = {
    "mjcf_scene":        "",     # "" -> bundled src/envs/assets/drone.xml
    "action_repeat":     2,      # 50 Hz control (timestep 0.01 * 2)
    "max_episode_steps": 250,    # ~5 s episodes
    "init_height":       2.0,    # spawn height (m) - descend from here to the pad
    "init_noise":        0.05,   # small random pose + velocity perturbation at reset
    "init_tilt":         0.2,    # gentler start tilt (rad) so the policy can converge
    "init_spin":         0.3,    # gentler start angular velocity (rad/s)
    "target_range_xy":   1.8,    # pad + obstacles sampled in x,y in [-1.8, 1.8] m
    "pad_z":             0.2,    # landing-pad height (the drone lands here)
    "thrust_gain":       1.0,    # action*gain about hover: action 0 = hover, +-1 = 0..2x hover
    "max_dist":          8.0,    # flew away this far = crashed
    "fail_tilt":         0.0,    # flipped past horizontal (up_z < 0) = crashed

    # PROCESS NOISE - the MACURA lever (gusts + actuator noise). Only in the REAL dynamics;
    # the ensemble learns them and its predictive spread IS the uncertainty MACURA acts on.
    "wind_force":        0.4,    # OU gust force std (N) ~9% of hover: real turbulence but recoverable
    "wind_correlation":  0.95,   # gust temporal correlation (smooth, sustained gusts) - keeps MACURA's edge
    "actuator_noise":    0.04,   # per-rotor multiplicative thrust noise (4% std)

    # OBSTACLES - virtual no-fly cylinders (analytic, so imagined rollouts see the same envelope)
    "n_obstacles":       2,
    "obstacle_radius":   0.4,    # crash if the drone's xy enters this radius of an obstacle
    "obstacle_min_clear": 0.7,   # keep obstacles clear of the start and pad at reset

    # LANDING / crash envelope
    "land_radius":       0.5,    # within this 3-D distance of the pad (at low speed, upright) = landed
    "soft_speed":        0.8,    # land softly below this speed (achievable under moderate wind)
    "impact_height":     0.06,   # below this height...
    "hard_speed":        1.5,    # ...moving faster than this = a hard crash (forgives light touchdowns)
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
    "gradient_steps_max": 4,        # Gmax in Eq. 22 (model-based UTD ceiling). Lowered 8->4: a high UTD
                                    # overtrains the critic on imagined data and drives the back-half churn
                                    # you saw; 4 is steadier and still 4x the model-free baseline. Also ~halves
                                    # model-based wall-clock, paying for the extra eval episodes above.
    "baseline_gradient_steps": 1,   # model-free SAC baseline UTD (~1 = standard SAC)
    "real_ratio": 0.1,              # within-batch real mixing (Janner/MBPO; MACURA inherits). Raised 0.05->0.1:
                                    # grounds the critic in twice as much REAL data -> less model exploitation.
}

# checkpoint selection & final eval (best = highest periodic greedy-eval return)
SELECTION = {"start_step": 1000, "eval_every": 1000, "final_eval_episodes": 30}

# ROLLOUT strategies (the ONLY thing that differs across algorithms).
# model_buffer_capacity is chosen so MACURA fills it (and thus reaches full adaptive UTD,
# Eq. 22) within roughly the first third of the run - NOT scaled linearly with
# TOTAL_ENV_STEPS: a too-large buffer would leave MACURA perpetually below MBPO/M2AC's fixed
# update budget. At 25k steps (capacity 12k) MACURA reaches |D_mod|_max around step ~7k.
ROLLOUT = {
    "freq_steps": 500, "num_rollouts": 200, "model_buffer_capacity": 12000,
    # MACURA: uncertainty-adaptive truncation (Algorithm 2)
    # xi is the ONE per-task knob (paper Table 5: xi in {0.3, 2, 5, 30} across envs; Tmax=10,
    # zeta=0.95 fixed). xi=1 is the paper's recommended starting point ("reasonable in all
    # environments", App. D.2). We use it here: xi=5 was too large for this windy drone task and
    # ENFORCED MODEL EXPLOITATION - MACURA barely truncated, behaved like MBPO@10, and the return
    # collapsed in the back half (peak ~step 13k then decayed, crash back to ~1.0). The paper's
    # own diagnosis (Sec 6.2 / App D.2): "instabilities due to model exploitation occur" for too-
    # large xi. Lower xi -> tighter kappa -> MACURA actually truncates uncertain rollouts -> stable.
    "macura": {"t_max": 10, "zeta": 0.95, "xi": 1.0, "adaptive_gradient_steps": True},
    # MBPO: fixed truncated-linear schedule; ramp scaled to REACH horizon 10 within the run.
    "mbpo": {"rollout_schedule": [1, 10, 500, 3000],
             "adaptive_gradient_steps": False, "fixed_gradient_steps": 4},  # matches SAC gradient_steps_max
    # M2AC: fixed length + mask least-trustworthy transitions. "ovr" = M2AC's own one-vs-rest
    # disagreement (paper-faithful); "gjs" = reuse MACURA's GJS (same-signal ablation).
    "m2ac": {"t_max": 10, "mask_fraction": 0.5, "uncertainty_penalty": 1.0,
             "uncertainty": "ovr", "fixed_gradient_steps": 4},  # matches SAC gradient_steps_max
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
