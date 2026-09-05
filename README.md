# MACURA learns a Backflip

**Teaching "Pogo" — a planar one-legged gymnast — to backflip, and using it to
compare the four algorithms from the MACURA paper (MACURA, MBPO, M2AC, SAC) on
an aggressive, unstable maneuver.**

> Paper: *Trust the Model Where It Trusts Itself — Model-Based Actor-Critic with
> Uncertainty-Aware Rollout Adaption* (MACURA), Frauenknecht, Eisele, Subhasish,
> Solowjow, Trimpe — **ICML 2024**, [arXiv:2405.19014](https://arxiv.org/abs/2405.19014).

Train on **Google Colab** (free GPU), then **watch the trained policy backflip
live on your Mac** in a native MuJoCo window.

---

## 1. Why a backflip

Model-based RL (MBRL) trains the policy on *imagined* rollouts from a learned
dynamics model, which is sample-efficient but risks **model exploitation**:
imagine too far and the model's errors compound, so the policy learns to exploit
fantasy dynamics.

**MACURA's idea** — *"trust the model where it trusts itself"* — makes the rollout
length **adaptive to the model's own uncertainty**: a branched rollout continues
while the probabilistic ensemble's members agree and **truncates the instant they
disagree**.

A **backflip is the ideal stress test**. It is an aggressive, unstable maneuver;
early in training the ensemble is very uncertain about the airborne/landing phase.
A **fixed-horizon** method (MBPO) keeps imagining the whole flight from a shaky
model, the policy exploits the fantasy, and Pogo **face-plants**. MACURA truncates
at the uncertain moment and learns the flip from trustworthy pieces — so it should
land the flip with fewer real crashes. This is not a paper benchmark (the paper
uses MuJoCo locomotion), so it is a genuine transfer to a new, unstable task.

### The four algorithms (shared SAC backbone + shared ensemble — only the rollout differs)

| Algorithm | Type | Rollout strategy | Role |
|---|---|---|---|
| **MACURA** | Model-based | **Adaptive** truncation (κ vs GJS divergence) | the proposed method |
| **MBPO** | Model-based | **Fixed** truncated-linear schedule | no-adaptivity baseline |
| **M2AC** | Model-based | fixed length, **mask** least-trustworthy steps | alternative uncertainty use |
| **SAC** | Model-free | **no model rollouts** | model-free reference |

---

## 2. The task (Pogo)

- **Body** ([`envs/assets/pogo.xml`](envs/assets/pogo.xml)): a planar one-legged
  pogo-stick with a little face. Motion is in the x–z plane; the root is two
  slide joints plus one **unlimited** hinge (the flip axis, free to rotate 360°),
  and three actuated leg hinges (hip / knee / ankle).
- **Observation (13):** `[torso_z, sin(θ), cos(θ), hip, knee, ankle, foot_clearance,
  x_vel, z_vel, θ_vel, hip_vel, knee_vel, ankle_vel]` — the torso angle enters as
  sin/cos so it is smooth across the 360° wrap.
- **Action (3):** normalized torque in `[-1,1]` on hip / knee / ankle.
- **Reward (dense, shaped, analytic):** alive bonus + foot-clearance (jump) +
  spin-while-airborne (the flip) + upright-while-grounded (land and hold) −
  control effort. It is a pure function of `(obs, action)`, so **imagined and real
  transitions are scored identically** — see `known_reward_fn` in
  [`envs/pogo_env.py`](envs/pogo_env.py). `w_rotation` is the curriculum knob.
- **Failure:** the torso dropping below `fail_torso_height` (a collapse/face-plant)
  ends the episode and is logged.
- **Success metric:** a full 360° rotation landed upright (`landed_flip`).

---

## 3. Repository structure

```
macura-backflip/
├── README.md
├── project.md / tasks.md          # idea + roadmap
├── requirements.txt
├── colab.ipynb                    # TRAIN here: setup -> study -> train 4 algos -> save -> plots -> videos
├── run_live_mac.py                # WATCH here: fly a saved policy live on macOS (mjpython)
├── bash/setup_colab.sh            # clone repo + pip install + Drive folders
├── configs/macura_backflip.yaml   # ALL hyperparameters (single source of truth)
├── envs/
│   ├── assets/pogo.xml            # the Pogo MuJoCo body
│   └── pogo_env.py                # make_env(), known_reward_fn(), termination_fn()
├── models/ensemble.py             # probabilistic ensemble + per-member Gaussians
├── algorithms/
│   ├── sac.py                     # shared SAC backbone
│   ├── macura.py                  # GJS uncertainty + adaptive-κ rollout (the method)
│   ├── mbpo.py                    # fixed truncated-linear rollout
│   └── m2ac.py                    # masking rollout
├── training/train.py              # train_one(algo, cfg, drive, seed), evaluate(...)
└── viz/
    ├── plots.py                   # env-study + comparison figures + video helpers
    └── live_viewer.py             # real-time macOS viewer (fly_policy / fly_sequence)
```

**Convention:** every `.py` is a **library of pure functions** — no top-level
execution. `colab.ipynb` orchestrates training; `run_live_mac.py` is the one
executable entry point (the local analogue of the notebook, for the Mac viewer).

---

## 4. How to run

### Train on Colab
1. **GitHub token** — this repo is private. Store a PAT in Colab Secrets as
   `GITHUB_TOKEN`, or enter it at the notebook's hidden prompt.
2. Open [`colab.ipynb`](colab.ipynb) in Colab (Runtime → GPU) and run top to
   bottom. `SMOKE = True` runs the whole notebook in minutes; `SMOKE = False`
   trains the real matrix (4 algorithms × seeds) and saves the best checkpoints,
   plots, and flight videos to Google Drive.

### Watch on your Mac
```bash
python -m pip install mujoco stable-baselines3 torch gymnasium pyyaml numpy imageio
# copy the best .zip checkpoints from Drive into ./runs/checkpoints/, then:
mjpython run_live_mac.py --ckpt runs/checkpoints/macura_seed0_best.zip
mjpython run_live_mac.py --compare runs/checkpoints        # fly all four in turn
```
Use `mjpython` (bundled with the `mujoco` pip package), not plain `python` —
MuJoCo's interactive viewer must own the main thread on macOS.

---

## 5. Expected result — honest

- **Sample efficiency** — MACURA/MBPO should reach a competent flip in fewer real
  steps than SAC.
- **Stability** — MACURA is expected to have the **highest stuck-backflip rate and
  lowest faceplant rate**, where fixed-horizon MBPO over-imagines the flight and
  crashes more.

This is a **faithful reproduction of the paper's mechanism on a new, unstable
task**, not a guaranteed win. A backflip is hard to discover from scratch, so the
config supports a **curriculum** (jump → half-flip → full flip). If, after a fair
run, MACURA does not win, that is reported honestly and diagnosed — numbers are
never doctored. See [`tasks.md`](tasks.md).

---

## 6. Renaming note

The internal project name is **macura-backflip**. To make the Colab clone work,
rename the GitHub repo to `macura-backflip` (GitHub → Settings → rename), or set
`REPO_NAME=<your-repo>` before running `bash/setup_colab.sh`. Renaming the local
folder (`mv macura-drone macura-backflip`) is optional/cosmetic.

---

## 7. References

- Frauenknecht et al., *MACURA*, ICML 2024 — [arXiv:2405.19014](https://arxiv.org/abs/2405.19014).
- Janner et al., *MBPO*, NeurIPS 2019. · Pan et al., *M2AC*, 2020. ·
  Haarnoja et al., *SAC*, 2018.
- Stable-Baselines3 — <https://github.com/DLR-RM/stable-baselines3>
- MuJoCo — <https://mujoco.org>
