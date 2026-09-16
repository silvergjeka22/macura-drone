# MACURA flies a Drone

A quadrotor learns to **recover from a tumbling start, fly to a target, and hold a stable
hover**, and that aggressive maneuver is used to compare the four algorithms of the MACURA
paper — **MACURA, MBPO, M2AC, SAC**. All four share one SAC backbone and one probabilistic
ensemble, so any difference is attributable to the **rollout strategy** alone. Train on a
**Kaggle GPU kernel** (pushed from VS Code, runs with your PC off), then watch the trained
policies fly on your Mac.

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
`bootstrap.setup()`, then `%run $ROOT/src/imports.py`, and follows a thin layout:

```
Get the code -> Setup -> Fresh start -> Meet the drone -> Train -> Results -> Flight demo -> Save
```

## Source layout

```
src/
  bootstrap.py            Kaggle: install deps, pick a safe render backend, make output dirs
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

## Run it on Kaggle from VS Code (PC off)

Everything runs on a Kaggle GPU kernel in batch mode: push the job from your terminal, turn
your PC off, and download the results later. There is nothing to keep open.

**One-time local setup**

1. Install the Kaggle CLI: `pip install kaggle`
2. Get an API token — on kaggle.com: Account → Settings → **Create New Token** (downloads
   `kaggle.json`). Put it where the CLI looks:
   - macOS/Linux: `~/.kaggle/kaggle.json` (then `chmod 600 ~/.kaggle/kaggle.json`)
   - Windows: `C:\Users\<you>\.kaggle\kaggle.json`
3. On Kaggle add a **Secret** named `GITHUB_TOKEN` (a GitHub personal-access token) so the
   kernel can clone this private repo. Internet + GPU are already enabled in
   `kernel-metadata.json`.

**Fill in two placeholders** (your Kaggle username):
`kernel-metadata.json` → `"id": "MYUSERNAME/macura-drone"` and `run.sh` → `KERNEL=...`.

**`kernel-metadata.json`** tells Kaggle how to run the notebook (JSON can't hold comments, so
the fields are explained here):

| field | meaning |
| --- | --- |
| `id` | `your-username/kernel-slug` — where the kernel lives on Kaggle |
| `code_file` | the notebook to run: `notebooks/drone.ipynb` |
| `language` / `kernel_type` | a Python notebook |
| `is_private` | keep the kernel private to you |
| `enable_gpu` | run on a GPU (needed for training) |
| `enable_internet` | allow the GitHub clone + pip installs |
| `dataset_sources` / `competition_sources` / `kernel_sources` | none — no external data needed |

**Push, check, download** (from the repo folder):

```bash
./run.sh push      # send the job to Kaggle and start it -> then turn your PC off
./run.sh status    # queued / running / complete
./run.sh get       # download results into ./out  (once status is complete)
```

Outputs are written to `/kaggle/working/runs` on the kernel and download into `./out`:
`checkpoints/` (best policies), `logs/` (per-run JSON), `plots/` (all figures). No
`/kaggle/input` dataset is needed — the drone model and code come from the repo and training
makes its own data. Budget: `config.TOTAL_ENV_STEPS` (40k) × `SEEDS` (5) × four algorithms;
for a quick health check first set `SEEDS=[0]`, `TOTAL_ENV_STEPS≈2000`, push, and confirm it
completes before the full run.

## Watch on your Mac

```bash
python -m pip install mujoco stable-baselines3 torch gymnasium numpy
# copy the best .zip checkpoints from ./out/runs/checkpoints into ./runs/checkpoints/, then:
mjpython run_live_mac.py --compare runs/checkpoints
```

## Honest expectation

Model-based RL beating model-free **SAC** on sample efficiency is the reliable win here.
MACURA beating **MBPO** on stability depends on the task being aggressive enough that MBPO
over-imagines — which is why the drone spawns tumbling. Numbers are never doctored: if MACURA
does not win after a fair run, that is reported and diagnosed.
