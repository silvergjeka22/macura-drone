import os

# Single source of truth for every quantity. Nothing is hard-coded in the other .py
# files; each receives the relevant sub-dict (ENV, ENSEMBLE, SAC, ROLLOUT...).
#
# TASKS: by default the CAGE task (land inside an open-top cage, see ENV). MACURA_TASK=delivery switches
# to the DELIVERY task (see the DELIVERY preset below the ENV dict). The four algorithms (MACURA, MBPO,
# M2AC, SAC) share the SAC backbone, the ensemble, pink-noise exploration and eval seeds - ONLY the
# rollout strategy differs.

# EXPERIMENT
# TASK, SEEDS, ALGORITHMS and the step budget are overridable from the environment (set them in a
# notebook cell BEFORE `from src.bootstrap import setup`, i.e. before config is first imported):
#   MACURA_TASK=delivery, MACURA_SEEDS="1 2 3", MACURA_ALGOS="macura mbpo", MACURA_STEPS=2000 (smoke test).
#   A Kaggle commit that runs past 12h is killed and saves NOTHING: at 8 updates/step, 4 algos take
#   ~3.4h per seed, so run at most 2-3 seeds per commit and merge the logs locally (train.load_runs).
def _env_list(name, default, cast):
    raw = os.environ.get(name, "")
    if not raw.strip():
        return default
    return [cast(x) for x in raw.replace(",", " ").split()]

SEED                = 0
SEEDS               = _env_list("MACURA_SEEDS", [0], int)   # PILOT: 1 seed to see whether the task separates
                                         # the methods before a full study (8-10 seeds for MACURA + MBPO).
ALGORITHMS          = _env_list("MACURA_ALGOS", ["macura", "mbpo", "m2ac", "sac"],
                                lambda x: x.strip().lower())
TOTAL_ENV_STEPS     = int(os.environ.get("MACURA_STEPS", "") or 40000)   # same as the earlier pilots.
                                         # ~3.4h per seed for all 4 algos -> 2 seeds per commit ~= 7h.
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
    # CAGE wall penalty (cage task): grows near the wall, only below its top (flying over is free)
    "w_cage":       0.8,
    "cage_scale":   0.15,  # distance scale (m) of the wall penalty
}
ENV = {
    "mjcf_scene":        "",     # "" -> bundled src/envs/assets/drone.xml
    "action_repeat":     2,      # 50 Hz control (timestep 0.01 * 2)
    "max_episode_steps": 400,    # 8 s episodes: a slow, careful descent into the cage under wind needs
                                 # ~4 s median and ~6 s for the slowest 10% (measured with the autopilot)
    "init_height":       2.0,    # spawn height (m) - descend from here to the pad
    "init_noise":        0.05,   # small random pose + velocity perturbation at reset
    "init_tilt":         0.2,    # gentler start tilt (rad) so the policy can converge
    "init_spin":         0.3,    # gentler start angular velocity (rad/s)
    "target_range_xy":   1.8,    # pad + obstacles sampled in x,y in [-1.8, 1.8] m
    "pad_min_dist":      1.7,    # min start->pad distance: keeps the drone's start OUTSIDE the ring
                                 # (ring reaches ~1.05 m from the pad) with room to approach the gap
    "pad_z":             0.2,    # landing-pad height (the drone lands here)
    "thrust_gain":       1.0,    # action*gain about hover: action 0 = hover, +-1 = 0..2x hover
    "max_dist":          8.0,    # flew away this far = crashed
    "fail_tilt":         0.0,    # flipped past horizontal (up_z < 0) = crashed

    # PROCESS NOISE - only in the REAL dynamics (reward/termination stay analytic).
    "wind_force":        0.4,    # OU gust std (N), ~9% of hover - same as the 40k ring baseline run
    "wind_correlation":  0.95,   # smooth, sustained gusts
    "actuator_noise":    0.04,   # per-rotor multiplicative thrust noise (4% std)

    # LANDING-ZONE PHYSICS - implemented in drone_env but OFF (all 0). Offline ensemble tests
    # (2026-09-23) showed each of them makes the 7-member ensemble AGREE MORE near the pad, not less:
    # GJS near/transit = turbulence 0.43x, ground effect 0.87x, deterministic wake 0.47-0.59x (vs 1.04x
    # without them) - the members widen their predicted variance where they can't fit, so MACURA would
    # trust the landing zone MORE. Kept as options for future tests, not used.
    "turb_force":        0.0,    # landing-zone turbulence std (N)
    "turb_correlation":  0.8,
    "turb_radius":       1.0,    # also the "landing zone" radius for MACURA's near-vs-transit diagnostic
                                 # (cage task: inside the cage + a margin)
    "turb_ramp":         0.15,
    "action_noise":      0.0,    # hidden additive action noise (paper App. D.4 method)
    "action_noise_zone": 0.0,
    "ge_gain":           0.0,    # ground effect
    "ge_scale":          0.25,
    "wake_gamma":        0.0,    # deterministic column wake (m^2/s)
    "pad_downwash":      0.0,    # deterministic pad downwash (m/s)

    # OBSTACLES - a RING of virtual no-fly columns AROUND the pad, with ONE entry gap facing the
    # start: the drone must thread the gap and land in the middle (analytic, so imagined rollouts
    # see the same envelope). This concentrates model uncertainty at the gap/pocket - exactly where
    # MBPO's fixed-horizon rollout over-imagines (clipping a column) and MACURA's truncation wins.
    "n_obstacles":       0,      # cage task: NO ring columns (the cage replaces them; obs = 14 dims)
    "obstacle_radius":   0.3,    # crash within this xy radius of a column (thinner -> threadable gap)
    "obstacle_min_clear": 0.5,   # keep columns clear of the start and each other at reset
    "ring_radius":       0.85,   # columns sit this far from the pad center (pocket radius ~0.55 m)
    "ring_gap_half_deg": 65.0,   # wider entry gap (~0.94 m opening) so landing is achievable while the
                                 # far columns still enclose the pad and punish MBPO's over-imagination

    # LANDING / crash envelope
    "land_radius":       0.5,    # within this 3-D distance of the pad (at low speed, upright) = landed
    "soft_speed":        0.8,    # land softly below this speed (achievable under moderate wind)
    "impact_height":     0.06,   # below this height...
    "hard_speed":        1.5,    # ...moving faster than this = a hard crash (forgives light touchdowns)
    # TOUCHDOWN option (OFF): the pad becomes a solid raised platform and success means actually
    # RESTING on it. Screened offline 2026-09-23 and REJECTED: contact makes the ensemble 19-26x more
    # WRONG at landing, but the members AGREE MORE there (GJS landing/transit 0.33x equal data, 0.76x
    # sparse; 0.92x / 1.38x without the platform) - MACURA would not detect it, both methods would suffer.
    "touchdown":         False,
    "pad_height":        0.15,   # platform top above the floor (m)
    "pad_radius":        0.35,   # platform radius (m) - same as the visual pad
    "touch_speed":       0.3,    # "resting on the pad" below this speed (m/s)
    # CAGE TASK (branch cage-env): the pad sits inside a circular cage of vertical bars, open at the
    # top. The drone spawns HIGH above the cage and must come in over the top and settle in the middle
    # without touching the wall. Crash = within cage_margin of the wall AND below its top (analytic).
    "cage":              True,
    "cage_radius":       0.8,    # cage wall radius around the pad center (m)
    "cage_height":       1.0,    # wall height (m) - flying over it is free, going through it is a crash
    "cage_margin":       0.15,   # ~drone half-span: crash if the drone center gets this close to the wall
    "cage_bars":         16,     # visual bars (rendering only)
    "start_height_min":  2.2,    # spawn height range (m): always ABOVE the cage
    "start_height_max":  2.6,
    "start_offset":      1.0,    # spawn up to 1 m sideways from the pad (sometimes outside the cage
                                 # footprint -> the drone has to come in over the wall)
    # DELIVERY TASK pieces - all OFF here; switched on together by the DELIVERY preset below.
    "spawn_above_pad":   False,  # spawn high above the pad (start_height_*, start_offset) without a cage
    "payload_max":       0.0,    # package weight (kg), new uniform draw every episode, IN the obs; 0 = off
    "vrs_loss":          0.0,    # fraction of thrust lost in a fast vertical descent; 0 = off
    "vrs_speed":         1.2,    # descent speed (m/s) where the loss starts ...
    "vrs_full":          2.2,    # ... and where it is complete
    "vrs_escape":        1.0,    # sideways speed (m/s) that flies out of it
    "vrs_powered":       False,  # delivery2: loss only when UPRIGHT + sinking along the rotor axis + THRUSTING
    "vrs_upright":       0.85,   # ... body-z world component where "upright" starts (~32 deg tilt)
    "vrs_thrust":        0.7,    # ... commanded thrust / hover thrust where "thrusting" starts
    "start_offset_min":  0.0,    # min spawn distance from the pad (m)
    "scenery":           False,  # windsock in the videos (visual only: no effect on physics or obs)
    "reward":            REWARD,
}

# DELIVERY TASK (opt-in: set MACURA_TASK=delivery before config is imported). A delivery drone starts
# ~2.4 m above the pad with a package of random, KNOWN weight and must land softly. The reward already
# pays every step spent settled on the pad, so arriving sooner is worth more (the "hurry" temptation).
# Descending straight down faster than ~1.2-2.2 m/s loses up to 35% of the thrust (a simplified vortex
# ring state), and a heavy package leaves less spare thrust to brake -> the safe speed depends on the
# weight. Measured with the hand-written autopilot (20 eval scenarios): descending at <= 1 m/s lands
# 100% at every weight; at 2 m/s it crashes 55% (empty) / 65% (0.1 kg) / 80% (0.2 kg). A 0.8-1.6 m/s
# threshold was too low: wind gusts alone pushed a careful empty drone into it (30% crashes).
# Meant for MACURA: slow-descent data never shows the lift loss, so a model trained on it may be
# confidently WRONG about fast, heavy descents - to be checked offline before any Kaggle run.
DELIVERY = {
    "cage": False, "n_obstacles": 0, "spawn_above_pad": True,
    "start_height_min": 2.2, "start_height_max": 2.6, "start_offset": 0.5,
    "payload_max": 0.2,          # up to +42% of the 0.48 kg drone: thrust/weight 2.0 (empty) .. 1.41 (full)
    "vrs_loss": 0.35, "vrs_speed": 1.2, "vrs_full": 2.2, "vrs_escape": 1.0,
    "turb_radius": 1.0,
    "scenery": True,             # windsock; videos also tint the drone red while it is losing thrust
}
# DELIVERY2 (opt-in: MACURA_TASK=delivery2) - built after the delivery pilot (2026-09-24), where the lift
# loss was NOT new to the models: early crashing / free-falling flights already sank fast, so the ensemble
# disagreed only 1.5x there and MACURA (filter too loose) was the one that learned to dive and crash.
#   * POWERED lift loss: only when the drone is upright, sinking along its rotor axis AND pushing thrust
#     (the real vortex ring state). Tumbling or free-falling drones never trigger it, so the danger first
#     shows up when a competent policy starts to hurry and brake - still new to the model at that point.
#   * Longer, higher approach: start 3-4 m up and 1-2 m to the side. Diving straight is fast but walks into
#     the lift loss; descending diagonally is safe (sideways speed escapes it) - a real strategy choice.
#   * Package weight stays visible; physics stays deterministic (random noise makes the members AGREE).
#   * HIGHER start, 5-6 m (was 3-4 m, changed 2026-09-24 BEFORE any delivery2 run): more room to build
#     speed makes hurrying pay more and the cliff steeper. Autopilot, 40 scenarios (seeds 5000-5039):
#                    3-4 m, 500 steps              5-6 m, 700 steps
#       0.55 m/s     lands 100%, return  869       lands 100%, return  993   (careful; p90 lands step 529)
#       1.0  m/s     lands  97%, return 1091       lands  97%, return 1438   (best: +26% -> +45% vs careful)
#       1.5  m/s     crash  55%, return  576       crash  70%, return  554
#       2.0  m/s     crash  65%, return  452       crash  88%, return  241   (dive)
DELIVERY2 = dict(DELIVERY, **{
    "vrs_powered": True, "vrs_upright": 0.85, "vrs_thrust": 0.7,
    "start_height_min": 5.0, "start_height_max": 6.0, "start_offset_min": 1.0, "start_offset": 2.0,
    "max_episode_steps": 700,    # 14 s: a careful 0.55 m/s descent from 6 m still lands with time to settle
})
TASK = os.environ.get("MACURA_TASK", "").strip().lower()
if TASK == "delivery":
    ENV.update(DELIVERY)
elif TASK == "delivery2":
    ENV.update(DELIVERY2)

# ENSEMBLE (probabilistic dynamics model, shared by all model-based algos)
ENSEMBLE = {
    "num_members": 5, "hidden_size": 200, "num_layers": 4, "activation": "silu",   # = baseline run (paper
                                                                                   # uses 7; kept at 5 so the
                                                                                   # pilot changes ONE thing)
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
    "gradient_steps_max": 8,        # Gmax in Eq. 22 (model-based UTD ceiling), SAME for all model-based (fair).
                                    # Raised 4->8: training HARD on imagined data is exactly where a fixed-
                                    # horizon method (MBPO) memorizes its model's mistakes and destabilizes,
                                    # while MACURA's truncated data stays clean (paper runs MBPO at G=10-30).
                                    # MACURA's adaptive UTD (Eq. 22) gives it slightly FEWER updates (~88%),
                                    # so this is conservative toward MACURA. SAC stays at 1.
    "baseline_gradient_steps": 1,   # model-free SAC baseline UTD (~1 = standard SAC)
    "real_ratio": 0.1,              # within-batch real mixing (Janner/MBPO; MACURA inherits). Raised 0.05->0.1:
                                    # grounds the critic in twice as much REAL data -> less model exploitation.
}

# checkpoint selection & final eval (best = highest periodic greedy-eval return). The final eval of the
# best checkpoint runs on FRESH scenarios (seeds 1000..1029), not on the selection scenarios (100..119):
# re-testing on the scenarios a checkpoint was picked on overrates lucky, high-variance checkpoints.
SELECTION = {"start_step": 1000, "eval_every": 1000, "final_eval_episodes": 30,
             "final_eval_seed_base": 1000}

# ROLLOUT strategies (the ONLY thing that differs across algorithms).
# model_buffer_capacity is chosen so MACURA fills it (and thus reaches full adaptive UTD,
# Eq. 22) within roughly the first third of the run - NOT scaled linearly with
# TOTAL_ENV_STEPS: a too-large buffer would leave MACURA perpetually below MBPO/M2AC's fixed
# update budget. At 25k steps (capacity 12k) MACURA reaches |D_mod|_max around step ~7k.
ROLLOUT = {
    "freq_steps": 500, "num_rollouts": 200, "model_buffer_capacity": 12000,
    # MACURA: uncertainty-adaptive truncation (Algorithm 2)
    # xi is the ONE per-task knob (paper Table 5: xi in {0.3, 2, 5, 30} across envs; Tmax=10,
    # zeta=0.95 fixed; App. D.2 recipe: start at 1, lower only if learning is unstable).
    # History on this task: xi=1 barely truncated (rollouts ~9/10 -> MACURA == MBPO); xi=0.4 truncated
    # (~8/10 late) but was timid early. Kept at 0.4 = the 40k baseline run, so the UTD pilot changes
    # ONE thing; under harder training the extra truncation is exactly the protection being tested.
    "macura": {"t_max": 10, "zeta": 0.95, "xi": 0.4, "adaptive_gradient_steps": True},
    # MBPO: fixed truncated-linear schedule; ramp scaled to REACH horizon 10 within the run.
    "mbpo": {"rollout_schedule": [1, 10, 500, 3000],
             "adaptive_gradient_steps": False, "fixed_gradient_steps": 8},  # matches SAC gradient_steps_max
    # M2AC: fixed length + mask least-trustworthy transitions. "ovr" = M2AC's own one-vs-rest
    # disagreement (paper-faithful); "gjs" = reuse MACURA's GJS (same-signal ablation).
    # uncertainty_penalty: with the corrected (non-negative KL) OvR the transitions M2AC keeps have u ~= 2
    # in the normalised model space vs ~1.9 reward/step, so 1.0 would cancel the whole imagined reward;
    # 0.1 makes it a mild ~10% pessimism. (The old 1.0 was set while the OvR was a negative "bonus".)
    "m2ac": {"t_max": 10, "mask_fraction": 0.5, "uncertainty_penalty": 0.1,
             "uncertainty": "ovr", "fixed_gradient_steps": 8},  # matches SAC gradient_steps_max
    "sac": {},
}

# EXPLORATION (kept CONSISTENT across algos to avoid the exploration confound)
EXPLORATION = {"type": "pink_noise", "scale": 0.3}

# PAPER PROTOCOL (applied with MACURA_TASK=delivery2): the MACURA paper's setup (App. D.1, Tables 4-6) and
# M2AC exactly as its own paper runs it (Pan et al. 2020, Alg. 2 + Sec. 5.1). The delivery pilot differed
# from the paper in ways that mostly hurt MACURA: ~1000x less imagined data (each imagined transition was
# reused ~460x), a 5-member ensemble without validation, and MACURA's filter effectively off.
#   * ensemble: 7 PNNs, validation split + early stopping, 5 elites for rollouts/uncertainty (mbrl-lib)
#   * imagined data: 100 rollouts per real step (25,000 every 250 steps; the paper uses ~400/step - a
#     quarter of it keeps a run inside Kaggle's 12 h while bringing the reuse down to paper level)
#   * SAC batches: 95% imagined / 5% real (paper), UTD 8 for MBPO/M2AC, Gmax 16 = 2x for MACURA (paper:
#     Eq. 22 uses about half of Gmax, so the actual update counts end up comparable)
#   * MACURA xi = 1, the paper's recommended starting point (App. D.2). The pilot's offline calibration
#     (0.12-0.15) does NOT transfer: its threshold was inflated by uncertainty spikes (running mean 12.6).
#     Measured with this protocol's 7/5-elite ensemble: xi 0.15 -> rollouts 0.1/10 (filter shuts MACURA
#     down), 0.5 -> ~2, 1.0 -> ~5, 1.5 -> ~6. The in-run running mean can still shift this, so xi is
#     TUNED (below). Diagnostic target: mean rollout 4-7 and low trust in fast dives. MACURA_XI overrides.
#   * M2AC: per-step masking w_h = (H-h)/(2(H+1)), one-vs-rest KL of the member that predicted, alpha=1e-3,
#     non-stop rollouts (its paper's defaults).
#   * exploration as in the MACURA paper: MACURA pink noise, MBPO/M2AC deterministic, SAC its own sampling.
#     This is a known confound (paper Fig. 5/11) and must be REPORTED; MACURA_EXPLORATION=equal gives
#     every algorithm the same pink noise instead.
# Tuning (fair: the same budget for MBPO): MACURA_XI in {0.5, 1, 2}, MBPO_HORIZON in {5, 10}; choose each
# method's value by its AVERAGE training-time eval return (selection scenarios 100-119), then report the
# full study on the fresh final-eval scenarios (1000+) only.
# (max_epochs 10 per model round - instead of open-ended early stopping - keeps a 4-algorithm seed < 12 h.)
if TASK == "delivery2":
    ENSEMBLE.update({"num_members": 7, "num_elites": 5, "holdout_ratio": 0.2, "holdout_max": 5000,
                     "max_epochs": 10, "patience": 3})
    ROLLOUT.update({"freq_steps": 250, "num_rollouts": 25000, "model_buffer_capacity": 1_000_000})
    ROLLOUT["macura"] = dict(ROLLOUT["macura"], xi=float(os.environ.get("MACURA_XI", "") or 1.0),
                             gradient_steps_max=16)
    _h = int(os.environ.get("MBPO_HORIZON", "") or 10)
    ROLLOUT["mbpo"] = dict(ROLLOUT["mbpo"], rollout_schedule=[1, _h, 500, 3000])
    ROLLOUT["m2ac"] = {"mode": "paper", "t_max": 10, "uncertainty_penalty": 1e-3, "uncertainty": "ovr",
                       "fixed_gradient_steps": 8}
    SAC["real_ratio"] = 0.05
    if os.environ.get("MACURA_EXPLORATION", "paper").strip().lower() != "equal":
        EXPLORATION["per_algo"] = {"macura": "pink_noise", "mbpo": "deterministic",
                                   "m2ac": "deterministic", "sac": "stochastic"}
ENV["task"] = TASK

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
