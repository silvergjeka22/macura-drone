# Architecture — `macura-drone`

**Goal:** demonstrate MACURA on an unstable quadrotor and compare all four paper algorithms
(MACURA, MBPO, M2AC, SAC). Trained on **Google Colab**, results mirrored to **Google Drive**.

## Convention
- **Every `.py` file is a library of plain functions** — no top-level execution, no `argparse`
  main. All orchestration happens in the **Colab notebook**, which imports these functions.
- **Training runs on Colab** (free GPU). The notebook clones this GitHub repo into the Colab
  workspace with a small bash script, imports the modules, and runs training.
- **Results (checkpoints, videos, plots) mirror to Google Drive** so they survive a session reset.

## Colab workflow
```
GitHub repo ──(git clone via bash)──> Colab workspace ──import──> notebook runs functions
                                            │
                                            └── results ──> Google Drive (mounted)
```
1. Mount Google Drive in the notebook.
2. Run `bash/setup_colab.sh` → clone the repo into `/content/macura-drone` (or `git pull`),
   `pip install -r requirements.txt`, create the Drive results folders, print the paths.
3. `import` the project's functions.
4. Call the training functions; checkpoints/videos/plots are written to Drive.

## Structure
```
macura-drone/
├── README.md
├── requirements.txt
├── colab.ipynb                 # the ONLY place code runs: clone → import → train → plot
├── bash/
│   └── setup_colab.sh          # clone repo into Colab + pip install + make Drive folders
├── configs/
│   └── macura_drone.yaml       # all hyperparameters (paper-faithful quantities)
├── envs/
│   └── drone_env.py            # make_env(cfg) → (env, obs_dim, act_dim)   [functions only]
├── models/
│   └── ensemble.py             # build_ensemble(), train_ensemble(), predict()
├── algorithms/
│   ├── sac.py                  # build_sac(), sac_update()    shared backbone
│   ├── macura.py               # macura rollout + adaptive-κ helpers
│   ├── mbpo.py                 # fixed-rollout variant
│   └── m2ac.py                 # masking variant
├── training/
│   └── train.py                # train_one(algo, cfg), evaluate(agent, cfg)   [functions]
└── viz/
    └── plots.py                # sample-efficiency curve, κ/rollout-depth, failure-rate plots
```

## Function-only contract (examples)
- `envs/drone_env.py`: `make_env(cfg, seed, render) -> (env, obs_dim, act_dim)`
- `models/ensemble.py`: `build_ensemble(cfg)`, `train_ensemble(ens, data)`, `predict(ens, s, a)`
- `algorithms/*.py`: `build_<algo>(obs_dim, act_dim, cfg)`, plus its rollout/update helpers
- `training/train.py`: `train_one(algo_name, cfg, drive_dir) -> run_dict`,
  `evaluate(agent, cfg, seeds) -> metrics`
- `viz/plots.py`: `plot_sample_efficiency(runs)`, `plot_rollout_depth(run)`

## Notebook sections (`colab.ipynb`)
1. Mount Drive + run `setup_colab.sh`.
2. Load `configs/macura_drone.yaml`.
3. Loop over the four algorithms × seeds → `train_one(...)` (saves to Drive).
4. `viz/plots.py` → the four-algorithm comparison figures.

## Conventions
- No execution in `.py` files — only definitions; the notebook drives them.
- One config YAML — all quantities there, nothing hard-coded.
- Drive mirroring — every checkpoint/video/plot copied to
  `/content/drive/MyDrive/macura-drone/...`.
- Reproducibility — fixed seeds from the notebook; same eval seeds across all four algorithms.
