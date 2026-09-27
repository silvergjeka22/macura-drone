# MACURA races a drone

Four reinforcement-learning algorithms learn the same MuJoCo drone race from scratch: **MACURA**, **MBPO**,
**M2AC** and **SAC**. They share the SAC learner, the world-model ensemble, the data and the evaluation
scenarios; only the way they use the learned model to imagine extra practice differs. MACURA stops each
imagined trip where its models disagree, so it does not trust its model everywhere.

> Paper: *Trust the Model Where It Trusts Itself: Model-Based Actor-Critic with Uncertainty-Aware Rollout
> Adaption*, Frauenknecht et al., ICML 2024 (arXiv:2405.19014, `paper/`).

Contents: [Layout](#layout) · [Run on Kaggle](#run-on-kaggle-one-seed-per-session) ·
[Compare all seeds](#compare-all-seeds-on-your-computer) · [Watch the drones on your Mac](#watch-the-drones-on-your-mac) ·
[What the results mean](#what-the-results-mean) · [Troubleshooting](#troubleshooting)

## Layout

```
notebooks/drone.ipynb     the Kaggle notebook: one seed of all four algorithms, results, trust, videos
compare.py                merge the downloaded seeds -> figures, tables, videos (your computer)
simulate.py               watch the trained drones race live in MuJoCo's 3-D viewer (Mac: mjpython)
run.sh, kernel-metadata.json   push the notebook to Kaggle / check / download
docs/EXPERIMENTS.md       why every setting has its value, earlier runs and their numbers
src/
  config/config.py        every setting (task presets, paper protocol); env vars MACURA_TASK, MACURA_SEEDS, ...
  envs/                   drone_env.py (the tasks), race_course.py, autopilot.py (hand-written, not learned), assets/
  models/ensemble.py      probabilistic ensemble world model
  algorithms/             sac.py (shared learner), macura.py, mbpo.py, m2ac.py
  training/               train.py (Dyna loop + evaluation), exploration.py, seed.py
  analysis/               results.py (loading, IQM + CIs, tables, model check), report.py (everything at once)
  viz/                    figures.py, video.py (race videos with MACURA's trust panel), simulator.py, scenery.py
```

## Run on Kaggle: the final study (one seed per session)

**Plan, decided before running:** seeds **1 and 2**, **100,000** real steps each, **MACURA, MBPO and SAC**, the
environment unchanged. One Kaggle session per seed, about 9 h each (~18 h of your GPU quota). Seed 0 (50,000
steps, all four algorithms) is the pilot; it is analysed on its own because curves of different lengths do not
mix. M2AC is not in the long runs: all four at 100k would need ~25 h and more than one 12 h session per seed.

Time per seed, estimated from the seed-0 Kaggle timings (the model fit grows with the data): MACURA ~4.5 h,
MBPO ~3.4 h, SAC ~0.5 h, setup + tests + videos ~0.6 h. Kaggle stops a session after 12 h and then saves
nothing, so the notebook stops training at 11 h (the run is cut there and saved) and skips videos when time is
short. It prints "session time limit" if that ever happens.

One-time setup on your computer:

1. `pip install kaggle`, then on kaggle.com: Settings → API → **Create New Token**; save `kaggle.json` to
   `~/.kaggle/kaggle.json` (`chmod 600 ~/.kaggle/kaggle.json`).
2. On Kaggle add a **Secret** named `GITHUB_TOKEN` (a GitHub token that can read this repo): notebook editor →
   Add-ons → Secrets. Phone verification is needed for internet and GPU.
3. Your username in `kernel-metadata.json` (`"id": "<username>/macura-drone"`) and in `run.sh` (`KERNEL=...`).

For each seed:

1. `git pull` (so `./run.sh push` uploads the current notebook).
2. In `notebooks/drone.ipynb`, first code cell: `SEED = 1` (then `SEED = 2` for the second session). Leave
   `STEPS = 100000` and `ALGOS = "macura mbpo sac"`.
3. `./run.sh push` (starts it on Kaggle; you can turn your computer off), `./run.sh status` until complete.
4. `./run.sh get` downloads into `./out`; rename it right away: `mv out out_seed1` (then `out_seed2`). Or on the
   notebook page: Output → Download, unzip into `out_seed1/`.

For seed 2 you can attach seed 1's output as an input (Add Input → Your Work): section 8 then compares both
seeds on Kaggle. Two accounts can also run the two seeds at the same time (each with its own username in
`kernel-metadata.json` / `run.sh` and its own `GITHUB_TOKEN` secret).

What the notebook shows for each algorithm, right after it is trained: its learning curve, the **test** of its
best checkpoint on 30 fresh scenarios (return, success, crash rate) and a **test video** of 3 of them (MACURA's
panel shows its world model's verdict at every step). At the end: all results, MACURA's trust figures, a race
video of the chosen algorithms (`SHOW`) and a 30 s MACURA vs MBPO flight.

Quick test first (optional, ~15 min): `STEPS = 2000`, push, check that it completes, then set it back.

Options (in the first code cell): `ALGOS` (which algorithms), and environment variables before setup:
`MACURA_EXPLORATION=equal` (the same pink noise for all), `MACURA_XI=1`, `MBPO_HORIZON=5`, `MBPO_UTD=16`.

## Compare the seeds on your computer

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt torch
python compare.py out_seed1 out_seed2 --out results             # figures + tables (~2 min)
python compare.py out_seed1 out_seed2 --out results --videos    # + race videos (~10 min)
python compare.py out_seed0 --out results_pilot                  # the 50k pilot (all four) on its own
```

It finds every `logs/<algo>_seed<n>.json` below the given folders (a seed found twice is used once; only the runs
of the longest length are compared, the others are listed as skipped) and writes:

| file | what |
|---|---|
| `results/summary.md` | scores per algorithm (2 seeds: their mean and each seed; 3+: IQM with 95% CI), per seed, trust numbers, autopilot reference |
| `plots/learning_curves.png` | return, crash rate, success, drones broken while learning (paper Fig. 4); rolling mean over 5 evaluations, raw values faint; shaded = the range of 2 seeds (95% CI with 3+) |
| `plots/summary_table.png` | the results table as a figure |
| `plots/scores.png` | the six scores as bars, dots = seeds |
| `plots/model_trust.png` | how much of its imagination each method trusts; MACURA in fast descents vs normal flight |
| `plots/kappa.png` | MACURA's threshold κ over training (paper Fig. 8) |
| `plots/model_check.png` | best world model on real test flights: disagreement vs actual error (paper Fig. 10) |
| `plots/untrusted_data.png` | imagined training data above MACURA's threshold; imagined fast descents |
| `videos/race_all.mp4` | the best policy of each algorithm on the same 3 fresh scenarios, MACURA's live trust panel |
| `videos/long_flight.mp4` | MACURA vs MBPO, 30 s demo flights |

## Watch the drones on your Mac

`simulate.py` opens MuJoCo's native 3-D viewer and races the trained drones **live, in real time**: every
algorithm flies its own copy of the exact training physics on the same scenario (same start, package, wind and
gusts), and all drones are drawn in one scene (MACURA blue, MBPO red, M2AC green, SAC grey), each with its trail.
The text panel shows laps, speed, sink rate and lift per drone and MACURA's world-model verdict at every step
("agree → trusted" or "DISAGREE → would stop imagining").

Setup (once), from the repo folder:

```bash
python3 -m venv .venv && source .venv/bin/activate      # Python 3 from python.org or Homebrew
pip install -r requirements.txt torch
```

`mjpython` comes with the `mujoco` package; macOS needs it instead of `python` for the viewer window.

Run (the `out_seed*` folders next to the repo files are found automatically):

```bash
mjpython simulate.py                                     # best seed of each algorithm, scenarios 1000-1004
mjpython simulate.py --runs out_seed1 out_seed2           # the 100k runs (the longest runs are used)
mjpython simulate.py --runs out_seed0                     # the 50k pilot, including M2AC
mjpython simulate.py --algos macura mbpo                 # only these two
mjpython simulate.py --seed 2                            # seed 2 of every algorithm instead of the best seed
mjpython simulate.py --seconds 30                        # 30 s flights (a demo: training flights are 10 s)
mjpython simulate.py --autopilot                         # add the hand-written autopilot at 1.5 and 3 m/s
mjpython simulate.py --follow MACURA --speed 0.5         # camera follows MACURA, half speed
mjpython simulate.py --scenarios 1000 1001               # which scenarios (1000+ = the fresh test scenarios)
```

In the window: **Space** pause/resume, **N** next scenario, **R** restart it; left-drag rotates the view,
right-drag moves it, scroll zooms; Tab / Shift+Tab show MuJoCo's side panels. The scenarios loop until you close the window; the result of each scenario
is printed in the terminal.

"Best seed" = the seed with the highest return on the selection scenarios (never chosen on the test scenarios).

## What the results mean

- **Main scores (fixed before the runs)**: average return while learning (how fast each method learns to race)
  and drones broken after the warm-up (real crashes while learning). Then, for the best checkpoint on 30 fresh
  scenarios: final test return, **success** and crash rate.
- **Success** = how much of a full lap (24.3 m) a drone flies in its 10 s flight, 0-100% (a crash ends the flight,
  so it counts how far it got). The earlier yes/no score, "a full lap inside one flight", needs more than 2.4 m/s
  on average and stayed 0 for every learner, so it could not tell them apart. Seed 0: MACURA 27%, M2AC 21%,
  MBPO 19%, SAC 6%; the autopilot 54% (1.5 m/s) and 95% (3 m/s).
- One evaluation is 20 flights, so single points are noisy: the curves show a rolling mean over 5 evaluations
  with the raw values faint behind. Every score is computed from the raw values.
- With 2 seeds the table shows their mean and each seed's value. If the two seeds disagree about which method is
  better, the study cannot tell them apart; say so rather than picking a winner. (3+ seeds: IQM with 95% CI.)
- Reference: the hand-written autopilot scores a return of 673 (1.5 m/s) and 1172 (3 m/s) on the selection
  scenarios, without crashes.
- **MACURA does not trust its model 100%**: `model_trust.png`, `kappa.png`, `model_check.png` and the trust panel
  in the videos and the simulator show where it stops imagining and whether the model is really more wrong there.
- Known confound (as in the paper): MACURA explores with pink noise, MBPO and M2AC deterministically. Details and
  the seed-0 numbers: `docs/EXPERIMENTS.md`.

## Troubleshooting

- `mjpython: command not found`: activate the venv where `mujoco` is installed (`source .venv/bin/activate`).
- `launch_passive requires that the Python script be run under mjpython`: start it with `mjpython`, not `python`.
- mjpython fails to start with a pyenv or conda Python: make the venv from python.org or Homebrew Python.
- `no run logs found`: pass the folders that contain `runs/logs` with `--runs`.
- A drone is missing: its `runs/checkpoints/<algo>_seed<n>_best.zip` was not downloaded.
- The HUD text does not show (old MuJoCo): `pip install -U mujoco`; the terminal still prints each scenario's result.
- Kaggle "session stopped after 12 h": one seed per session (the notebook's `SEED`), 50,000 steps.
