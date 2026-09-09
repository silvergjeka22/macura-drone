# MACURA flies a Drone

A quadrotor learns to **recover from a tumbling start, fly to a target, and hold a stable
hover**, and that aggressive maneuver is used to compare the four algorithms of the MACURA
paper — **MACURA, MBPO, M2AC, SAC**. All four share one SAC backbone and one probabilistic
ensemble, so any difference is attributable to the **rollout strategy** alone. Train on
Google Colab (GPU), then watch the trained policies fly on your Mac.

> Paper: *Trust the Model Where It Trusts Itself — Model-Based Actor-Critic with
> Uncertainty-Aware Rollout Adaption*, ICML 2024 (arXiv:2405.19014).

## Why a drone

- **Smooth, learnable flight dynamics** → the ensemble fits them well, so model-based RL is
  very sample-efficient (this is where MACURA/MBPO/M2AC beat model-free **SAC**).
- **Aggressive recovery** (high angular rates after a tumbling start, fast approach and hard
  deceleration) → the learned model is most *uncertain* there, so a **fixed-horizon** rollout
  (MBPO) over-imagines and destabilizes, while **MACURA** truncates the imagined rollout the
  instant the ensemble disagrees. That is MACURA's stability advantage.

## The task

- **Body:** a Skydio-X2-style quadrotor (custom MuJoCo model, `src/envs/assets/drone.xml`) —
  a free-flying rigid body with 4 rotor thrust motors.
- **Observation (14-dim):** target-relative position, height, orientation quaternion, and
  linear + angular velocity.
- **Action (4-dim):** normalized thrust per rotor, centered on hover (0 = hold altitude).
- **Reward (dense, analytic in obs+action):** be at the target (`exp(-dist)`), stay level,
  low spin, low speed, minus control effort — identical for real and imagined transitions.
- **Reset:** the drone spawns **tilted and tumbling** (`init_tilt` / `init_spin`).
- **Crash:** hitting the ground or flying away ends the episode.

## Notebook

`notebooks/drone.ipynb` — the one place training runs. It clones `src/`, runs
`bootstrap.setup()`, then `%run src/imports.py`, and follows a thin layout:

```
Connect Colab -> Setup -> Fresh start -> Meet the drone -> Train -> Results -> Flight demo -> Save
```

## Source layout

```
src/
  bootstrap.py            Colab: mount Drive, install deps, pick a safe render backend
  imports.py              load every symbol into the notebook namespace (%run target)
  config/config.py        every quantity: env, reward, ensemble, sac, rollout, exploration
  envs/
    drone_env.py          make_env, known_reward_fn, termination_fn (the analytic reward)
    assets/drone.xml      the quadrotor MuJoCo body (self-contained, no meshes)
  models/ensemble.py      probabilistic ensemble + per-member Gaussians
  algorithms/
    sac.py                shared SAC backbone + within-batch real/imagined mixing
    macura.py             GJS uncertainty + adaptive-kappa rollout (the method)
    mbpo.py               fixed truncated-linear rollout
    m2ac.py               fixed-length rollout + one-vs-rest masking
  training/
    seed.py               set_seed
    train.py              train_one, evaluate, evaluate_best, record_best_videos
  viz/
    plots.py              comparison figures + video helpers
    live_viewer.py        real-time macOS viewer (fly_policy / fly_sequence)
run_live_mac.py           watch a saved policy fly on macOS (mjpython)
```

Every `.py` under `src/` is a library of pure functions; the notebook orchestrates them,
and `run_live_mac.py` is the one executable entry point.

## Running on Colab

Open `notebooks/drone.ipynb`, select a **GPU** runtime, and run top to bottom: it clones
`src/` from GitHub, `bootstrap.setup()` (mount Drive, install, pick a render backend), then
`%run src/imports.py`. Best checkpoints, plots and logs go to Drive under `config.DRIVE_ROOT`.
Push before running, since Colab clones `src/` from GitHub. The training budget is
`config.TOTAL_ENV_STEPS` × `SEEDS` × the four algorithms (raise the steps for a decisive run).

## Watch on your Mac

```bash
python -m pip install mujoco stable-baselines3 torch gymnasium numpy
# copy the best .zip checkpoints from Drive into ./runs/checkpoints/, then:
mjpython run_live_mac.py --compare runs/checkpoints
```

## Honest expectation

Model-based RL beating model-free **SAC** on sample efficiency is the reliable win here.
MACURA beating **MBPO** on stability depends on the task being aggressive enough that MBPO
over-imagines — which is why the drone spawns tumbling. Numbers are never doctored: if MACURA
does not win after a fair run, that is reported and diagnosed.
