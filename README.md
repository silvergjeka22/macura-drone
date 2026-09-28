# MACURA races a drone

Four reinforcement-learning algorithms learn the same MuJoCo drone race from scratch: **MACURA**, **MBPO**,
**M2AC** and **SAC**. They share the SAC learner, the world-model ensemble, the data and the evaluation
scenarios; only the way they use the learned world model to imagine extra practice differs. MACURA stops each
imagined trip where its models disagree, so it does not trust its model everywhere.

> Paper: *Trust the Model Where It Trusts Itself: Model-Based Actor-Critic with Uncertainty-Aware Rollout
> Adaption*, Frauenknecht et al., ICML 2024 (arXiv:2405.19014, `paper/`).

**The task (race3):** laps around a 6 x 6 m course with four gates, two pillars and a near-vertical 3 m chute, under
wind, gusts and a package of random weight. Sinking too fast in the chute loses up to 50% of the lift and crashes the
drone; the learned world models are wrong there and disagree about it, which is what MACURA can detect.

## The pipeline

```
training/   4 identical Kaggle notebooks (built from one template), 2 per account, all running at the same time
              macura_mbpo_seeds45   MACURA + MBPO, seeds 4 5   account nouradon
              m2ac_sac_seeds45      M2AC + SAC,    seeds 4 5   account nouradon
              macura_mbpo_seeds67   MACURA + MBPO, seeds 6 7   account silvergjeka01
              m2ac_sac_seeds67      M2AC + SAC,    seeds 6 7   account silvergjeka01
            each one: training (live printout of every run), results, figures, MACURA's trust in its model,
            a 30 s video of each seed's best checkpoint, and one results file <name>.zip
kaggle.sh   push / status / get for the 4 notebooks (get -> downloads/<name>, never overwrites)
testing/    test.ipynb, on your computer: all training results together, a test of every model on 50 new
            scenarios with plots and videos, the final comparison with the best results plot, a check that every model opens in the simulator
simulate.py watch the trained drones race live in MuJoCo's 3-D viewer (macOS: mjpython)
```

Step by step (accounts, keys, the one-time Kaggle setup, troubleshooting): **[HOW_TO_RUN.md](HOW_TO_RUN.md)**.

```bash
./kaggle.sh push                                  # start the 4 training notebooks
./kaggle.sh status                                # until all 4 are COMPLETE (about 4-5 h)
./kaggle.sh get                                   # download them into downloads/
jupyter notebook testing/test.ipynb               # Run All -> results/ (tables, figures, videos)
mjpython simulate.py                              # watch them race
```

## Settings (decided before the runs)

- race3, **50,000 real steps** per run (5,000 random warm-up + 45,000 learning), **seeds 4-7** (0-3 were the race3 test), the MACURA paper's
  protocol (7-network ensemble, 25,000 imagined trips every 250 steps, 95% imagined / 5% real data per SAC batch).
- **Equal training updates**: MBPO and M2AC do 12 SAC updates per real step = MACURA's average (it adapts, up to 16).
- Exploration as in the paper (disclosed): MACURA pink noise, MBPO and M2AC without noise, SAC samples its policy.
- Best checkpoint chosen on scenarios 100-119, tested on 1000-1029 in training and on new ones (3000-3049) in
  `testing/test.ipynb`; nothing is tuned on test scenarios.
- **Lap progress** = how much of a full lap (24.3 m) a 10 s flight covers, in %; a crash ends the flight.

Why every setting has its value, every earlier run and its numbers: `docs/EXPERIMENTS.md`.

## Code

```
src/config/config.py      every setting; env vars MACURA_STEPS, MACURA_SEEDS, MBPO_UTD, M2AC_UTD, ...
src/envs/                 drone_env.py (race3), race_course.py, autopilot.py (hand-written, not learned), assets/
src/models/ensemble.py    probabilistic ensemble world model
src/algorithms/           sac.py (shared learner, model loading), macura.py, mbpo.py, m2ac.py
src/training/             train.py (Dyna loop + evaluation), exploration.py, seed.py
src/analysis/             results.py (loading, IQM + CIs, tables, model check), testing.py (new-scenario test),
                          report.py (all figures at once)
src/viz/                  figures.py, test_figures.py, video.py (race videos with MACURA's trust), simulator.py
```

Only the race3 task is in the code. The earlier tasks (cage, delivery, race, race2) and their runs are described in
`docs/EXPERIMENTS.md`; their code is in the git history (last version with all tasks: commit `e402d16`).
