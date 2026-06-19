# MACURA on Drones

**Reproducing and stress-testing the MACURA algorithm on an unstable quadrotor,
and comparing all four algorithms from the paper (MACURA, MBPO, M2AC, SAC).**

> Paper: *Trust the Model Where It Trusts Itself — Model-Based Actor-Critic with
> Uncertainty-Aware Rollout Adaption* (MACURA), Frauenknecht, Eisele, Subhasish,
> Solowjow, Trimpe — **ICML 2024**, [arXiv:2405.19014](https://arxiv.org/abs/2405.19014).

---

## 1. Project overview

Model-based RL (MBRL) is sample-efficient because the agent trains on *imagined*
rollouts from a learned dynamics model instead of expensive real interaction. The
risk is **model exploitation**: imagine too far and the model's small errors
compound, so the policy learns to exploit fantasy dynamics.

**MACURA's idea** — *"trust the model where it trusts itself"* — makes the rollout
length **adaptive to the model's own uncertainty**: a branched rollout continues
while the probabilistic ensemble's members agree (low epistemic uncertainty) and
**truncates the instant they disagree**.

This project transfers that mechanism to a domain *not* in the paper — an
**unstable Skydio X2 quadrotor** (MuJoCo) — where model errors compound fast, so
adaptive truncation is put under real stress. We run **all four** paper
algorithms with a shared SAC backbone and shared ensemble so any difference is
attributable to the **rollout strategy alone**.

### The four algorithms

| Algorithm | Type | Rollout strategy | Uncertainty use | Role |
|---|---|---|---|---|
| **MACURA** | Model-based | **Adaptive** truncation (per-rollout, κ vs GJS divergence) | drives an adaptive κ threshold | the proposed method |
| **MBPO** | Model-based | **Fixed** truncated-linear schedule | none | no-adaptivity baseline |
| **M2AC** | Model-based | fixed length, **mask** least-trustworthy steps | filter, not length | alternative uncertainty use |
| **SAC** | Model-free | **no model rollouts** | — | model-free reference |

---

## 2. Core concepts (as implemented)

- **SAC backbone** — soft actor-critic, identical for all four (built on
  `mbrl-lib`'s SAC). See [`algorithms/sac.py`](algorithms/sac.py).
- **Probabilistic ensemble (PE)** — `mbrl-lib` `GaussianMLP`, predicts next-state
  Gaussians; member disagreement is the uncertainty signal. Log-variances are
  bounded for numerically stable uncertainty. See [`models/ensemble.py`](models/ensemble.py).
- **GJS uncertainty** — geometric Jensen–Shannon divergence between member
  Gaussians (paper Eq. 15–19), closed-form for diagonal covariances. Implemented
  exactly in [`algorithms/macura.py`](algorithms/macura.py) (`compute_gjs`).
- **Adaptive threshold κ** — running mean of `ξ · (ζ-quantile of first-step
  uncertainties)` (Eq. 21); `update_kappa`.
- **Update-to-data scaling** — SAC gradient steps scale with model-buffer
  fullness (Eq. 22); `gradient_steps`.
- **Known reward + termination** — analytic dense quadratic reward and the
  failure envelope are shared by all model-based rollouts for fairness. See
  `known_reward_fn` / `termination_fn` in [`envs/drone_env.py`](envs/drone_env.py).

### The drone task (low-step by design)
- **Env:** Skydio X2 (MuJoCo, from `mujoco_menagerie`), hover at `(0,0,1)` m;
  optional path-following.
- **Observation (13):** `[pos_error(3), quaternion(4), linear_vel(3), angular_vel(3)]`.
- **Action (4):** per-rotor thrust in `[-1,1]`, mapped to the model's ctrl ranges.
- **Reward (dense quadratic):** penalizes position error, velocity, tilt, angular
  velocity, control effort; plus an alive bonus.
- **Three low-step levers:** **near-hover initialization**, **action repeat
  (frame-skip)**, and a **smooth dense reward** — so meaningful curves emerge in
  ~tens of thousands of *real* steps.

---

## 3. Repository structure

```
macura-drone/
├── README.md                  # this file
├── tasks.md                   # actionable roadmap / checklist
├── requirements.txt           # dependencies (Colab-targeted)
├── colab.ipynb                # THE place code runs: clone → env study → train → plot
├── bash/
│   └── setup_colab.sh         # clone repo + Skydio X2 + install + Drive folders
├── configs/
│   └── macura_drone.yaml      # ALL hyperparameters (single source of truth)
├── envs/
│   └── drone_env.py           # make_env(), known_reward_fn(), termination_fn()
├── models/
│   └── ensemble.py            # build/train/predict + per-member Gaussians
├── algorithms/
│   ├── sac.py                 # shared SAC backbone
│   ├── macura.py              # GJS uncertainty + adaptive-κ rollout (the method)
│   ├── mbpo.py                # fixed truncated-linear rollout
│   └── m2ac.py                # masking rollout
├── training/
│   └── train.py               # train_one(algo, cfg, drive, seed), evaluate(...)
└── viz/
    └── plots.py               # environment-study + comparison figures
```

**Convention:** every `.py` is a **library of pure functions** — no top-level
execution, no argparse. The Colab notebook imports and orchestrates them.

---

## 4. Installation & setup (Google Colab)

Training runs on **Colab** (free GPU); results mirror to **Google Drive**.

1. **GitHub token** — this repo is private. Create a PAT with `repo` scope and
   store it in **Colab Secrets** as `GITHUB_TOKEN` (recommended), or enter it at
   the notebook's hidden `getpass` prompt. **Never commit or paste it into code.**
2. Open [`colab.ipynb`](colab.ipynb) in Colab (Runtime → GPU).
3. Run cells **0–3**: mount Drive → load token → run `bash/setup_colab.sh`
   (clones this repo + `mujoco_menagerie` for the Skydio X2, installs
   `requirements.txt`, creates Drive folders) → load the config.

> The setup script reads `GITHUB_TOKEN` from the environment for a single clone
> and scrubs it from the git remote afterward — it is never written to disk.

### Local (optional, env study only)
```bash
python -m pip install mujoco gymnasium numpy matplotlib pyyaml imageio
# then import envs.drone_env / viz.plots in a Python session
```

---

## 5. How to run

**Part 1 — environment study (ready to run):** notebook cells 4–10 create the
env and produce the study figures (random-policy instability, free dynamics,
state distribution, reward landscape, action response, a rendered frame). These
need only MuJoCo + Gymnasium, so they validate the env before any training.

**Part 2 — four-algorithm comparison:**
```python
from training import train as trainer
# smoke test first (short!)
run = trainer.train_one('macura', cfg_smoke, DRIVE, seed=0)
# then the full matrix
runs = [trainer.train_one(a, cfg, DRIVE, seed=s)
        for a in cfg['experiment']['algorithms']
        for s in cfg['experiment']['seeds']]
```
Then the comparison figures:
```python
from viz import plots
plots.plot_sample_efficiency(runs)   # return vs real steps, mean ± std
plots.plot_failure_rate(runs)        # crashes while learning
plots.plot_final_quality(runs)       # best policy per algorithm
plots.plot_rollout_depth(macura_run) # MACURA signature: κ & rollout length
```

All hyperparameters live in [`configs/macura_drone.yaml`](configs/macura_drone.yaml).
Start with `total_env_steps` small to validate the pipeline, then scale up.

---

## 6. Expected results

- **Sample efficiency** — MACURA reaches a target return in fewer *real* steps
  than SAC; model-based methods take off faster.
- **Stability** — on the unstable drone, fixed-horizon **MBPO is expected to
  wobble/collapse** where MACURA's adaptive truncation stays stable.
- **Signature figure** — MACURA's rollout length is short early (uncertain model)
  and grows as the model becomes reliable; κ decreases over training.

> This is a faithful **reproduction + transfer**, not a claim to beat the paper's
> numbers. A mixed/negative result on the drone is still a valid finding.

---

## 7. Honest caveats

- **Compute, not samples, is the bottleneck on Colab.** MBRL is sample-efficient
  but does heavy ensemble retraining + many SAC updates per step. "Few steps" ≠
  "fast wall-clock." Checkpoint/resume is built in for this reason.
- **Fairness is everything.** Shared backbone/ensemble/seeds/eval; one consistent
  exploration and UTD policy. See `tasks.md` for the traps to control.
- **mbrl-lib version sensitivity.** The `models/ensemble.py`, `algorithms/sac.py`
  wrappers follow `mbrl-lib`'s documented API; the first Colab task is to confirm
  the import/signature chain on the installed version.
- **MuJoCo has no built-in drone task** — we adapt the Skydio X2 MJCF; rotor
  aerodynamics are simplified (smoother and easier to learn, which suits a
  method reproduction).

---

## 8. References

- Frauenknecht et al., *MACURA*, ICML 2024 — [arXiv:2405.19014](https://arxiv.org/abs/2405.19014).
  Authors' code: <https://github.com/Data-Science-in-Mechanical-Engineering/macura>
- Janner et al., *MBPO*, NeurIPS 2019. · Pan et al., *M2AC*, 2020. ·
  Haarnoja et al., *SAC*, 2018.
- `mbrl-lib`: Pineda et al., 2021 — <https://github.com/facebookresearch/mbrl-lib>
- `mujoco_menagerie` (Skydio X2) — <https://github.com/google-deepmind/mujoco_menagerie>
