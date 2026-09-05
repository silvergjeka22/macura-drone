# MACURA learns a Backflip

Teaching **Pogo** — a planar one-legged pogo-stick gymnast — to backflip, and using
that aggressive, unstable maneuver to compare the four algorithms of the MACURA
paper (**MACURA, MBPO, M2AC, SAC**). All four share one SAC backbone and one
probabilistic ensemble, so any difference is attributable to the rollout strategy
alone. Train on Google Colab (GPU), then watch the trained policies fly live on your
Mac.

> Paper: *Trust the Model Where It Trusts Itself — Model-Based Actor-Critic with
> Uncertainty-Aware Rollout Adaption*, ICML 2024 (arXiv:2405.19014).

## Why a backflip

A backflip is where a learned model is most uncertain — the aggressive airborne and
landing phase. A **fixed-horizon** rollout (MBPO) keeps imagining the whole flight
from a shaky model, the policy exploits the fantasy, and Pogo face-plants. **MACURA**
truncates the rollout the instant the ensemble disagrees and learns the flip from
trustworthy pieces. It is not a paper benchmark (the paper uses MuJoCo locomotion),
so it is a genuine transfer to a new, unstable task.

## Notebook

`notebooks/backflip.ipynb` — the one place training runs. It clones `src/`, runs
`bootstrap.setup()`, then `%run src/imports.py`, and follows the same thin layout:

```
Connect Colab  ->  Setup  ->  Meet Pogo  ->  Train  ->  Results  ->  Flight demo  ->  Save
```

## Source layout

```
src/
  bootstrap.py            Colab: mount Drive, install deps, make result folders
  imports.py              load every symbol into the notebook namespace (%run target)
  config/config.py        every quantity: env, reward, ensemble, sac, rollout, exploration
  envs/
    pogo_env.py           make_env, known_reward_fn, termination_fn (the analytic reward)
    assets/pogo.xml       the Pogo MuJoCo body
  models/ensemble.py      probabilistic ensemble + per-member Gaussians
  algorithms/
    sac.py                shared SAC backbone
    macura.py             GJS uncertainty + adaptive-kappa rollout (the method)
    mbpo.py               fixed truncated-linear rollout
    m2ac.py               masking rollout
  training/
    seed.py               set_seed
    train.py              train_one, evaluate, evaluate_best, record_best_videos
  viz/
    plots.py              comparison figures + video helpers
    live_viewer.py        real-time macOS viewer (fly_policy / fly_sequence)
run_live_mac.py           watch a saved policy fly on macOS (mjpython)
```

Every `.py` is a library of pure functions; the notebook orchestrates them, and
`run_live_mac.py` is the one executable entry point.

## Running on Colab

Each run opens with the same cells: clone `src/` from GitHub, `bootstrap.setup()`
(mount Drive, install, make folders), then `%run src/imports.py`. The best
checkpoints, plots, and flight videos go to Drive under `config.DRIVE_ROOT`. Push
before running, since Colab clones `src/` from GitHub. Rename the GitHub repo to
`macura-backflip` (or set `REPO`/`REPO_NAME`) so the clone resolves.

## Watch on your Mac

```bash
python -m pip install mujoco stable-baselines3 torch gymnasium numpy
# copy the best .zip checkpoints from Drive into ./runs/checkpoints/, then:
mjpython run_live_mac.py --compare runs/checkpoints
```

## Honest expectation

This is a faithful reproduction of the paper's mechanism on a new, unstable task,
not a guaranteed win. A backflip is hard to discover from scratch, so `w_rotation`
in `config.py` is a curriculum knob (0 = jump & balance → 1 = full flip). If MACURA
does not win after a fair run, that is reported and diagnosed — numbers are never
doctored. See `tasks.md`.
