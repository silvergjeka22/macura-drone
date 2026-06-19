# Project 1 — MACURA on Drones

**Reproducing and extending the MACURA paper on a quadrotor control task, and comparing all
four algorithms studied in the paper.**

> Paper: *Trust the Model Where It Trusts Itself — Model-Based Actor-Critic with
> Uncertainty-Aware Rollout Adaption*, Frauenknecht, Eisele, Rastogi, Trimpe et al.,
> ICML 2024 (arXiv:2405.19014).

This document describes the **project idea only** — the task, the goal, the methodology as
defined in the paper, and the four-algorithm comparison. No code.

---

## 1. The idea in one paragraph

Model-based reinforcement learning (MBRL) is sample-efficient because the agent learns a model
of the environment and trains its policy on *imagined* rollouts instead of expensive real
interactions. The danger is **model exploitation**: if you imagine too far ahead, the model's
small errors compound and the policy learns to exploit fantasy dynamics. The MACURA paper's
contribution is to make the rollout length **adaptive to the model's own uncertainty** — "trust
the model where it trusts itself." This project takes that exact mechanism and applies it to a
domain *not* in the paper: controlling an **unstable quadrotor**, where model errors compound
especially fast, so the adaptive-truncation idea is put under real stress.

## 2. Why a drone is the right test for MACURA

The paper validates on MuJoCo locomotion (smooth, fairly stable systems). A quadrotor is a
deliberately harder and more revealing case for MACURA's central claim:

- **Dynamics are smooth and learnable** → the probabilistic ensemble can fit them, so MBRL is
  viable (unlike sparse-reward tasks, where MBRL fails).
- **The platform is unstable** → it balances on thrust, so a small model error grows within a
  few steps. This is precisely the regime where a *fixed* rollout horizon over-imagines and a
  policy degrades — and where MACURA's *adaptive* truncation should shine.
- **Sample efficiency has a real meaning** → real drone data is expensive and crashes are
  literal, so "fewer real interactions to learn" is a genuine safety/cost win, not just a curve.

The thesis-level message: *MACURA's adaptive rollout truncation is not a marginal trick — on an
unstable system it is what keeps model-based learning stable, and it gets there with fewer real
samples than the model-free baseline.*

## 3. The task

- **Environment:** a quadrotor simulator (e.g. `gym-pybullet-drones`), starting with the
  **hover / stabilisation** task and then a **trajectory-tracking** task (waypoints / figure-eight).
- **Observation:** the drone's kinematic state — position, orientation, linear and angular
  velocities.
- **Action:** continuous low-level motor commands (4 rotor inputs).
- **Reward:** **dense** stabilisation / tracking reward (distance to target pose, penalty for
  drift and excessive control). Dense reward is required — the paper's method, and MBRL in
  general, relies on a reward the learned model can predict accurately.
- **Termination as failure:** tumbling or leaving the safe flight envelope ends the episode and
  is recorded as a failure (so a policy cannot look "safe" by crashing out early).

## 4. The goal

1. **Reproduce the paper's mechanism faithfully** on the drone task (the four algorithms below,
   sharing one SAC backbone and one dynamics ensemble, exactly as the paper structures them).
2. **Show MACURA's two claimed advantages** in this new domain:
   - **Sample efficiency** — reach a target return in fewer real environment steps than the
     model-free baseline (SAC).
   - **Stability** — avoid the model-exploitation collapse that fixed-horizon model-based
     methods (MBPO) suffer on an unstable system.
3. **Demonstrate the adaptive mechanism directly** — show that the rollout length adapts down
   when the model is uncertain (early training, aggressive manoeuvres) and grows as the model
   becomes reliable.

## 5. Methodology (as defined in the paper)

All four agents share the **same** components so the comparison is fair — only the rollout
strategy and exploration differ:

- **SAC backbone** — soft actor-critic (twin critics, automatic temperature, tanh-squashed
  Gaussian policy) is the learner in every variant.
- **Probabilistic dynamics ensemble** — several Gaussian neural networks predict the next-state
  and reward distribution; ensemble *disagreement* is the uncertainty signal.
- **Uncertainty measure** — MACURA quantifies model uncertainty with the **generalized
  Jensen–Shannon divergence (GJS)** between the ensemble members' predictive distributions.
- **Adaptive rollout threshold κ** — a running statistic (the ζ-percentile of first-step
  uncertainties, scaled by a factor ξ, averaged over recent rounds). A rollout from a real state
  continues while its uncertainty stays below κ and stops otherwise, up to a maximum length.
- **Dyna training loop** — periodically retrain the ensemble on real data, generate fresh
  imagined rollouts into a model buffer, and update SAC from a mix of real and imagined data;
  the number of gradient updates scales with how full the model buffer is.

## 6. The four-algorithm comparison (the core of this project)

The paper studies **four** algorithms. This project runs all four on the drone task with shared
backbone/ensemble, identical seeds, and identical evaluation, so any difference is attributable
to the rollout/uncertainty strategy alone.

| Algorithm | Type | Rollout strategy | Uncertainty use | Role in the comparison |
|---|---|---|---|---|
| **MACURA** | Model-based | **Uncertainty-adaptive** truncation (length set per-rollout by κ vs GJS divergence) | GJS divergence drives an adaptive κ threshold | **The proposed method** — expected best sample efficiency *and* stability |
| **MBPO** | Model-based | **Fixed** truncated-linear schedule (length ramps over training, same for all rollouts) | None | Shows what happens *without* adaptivity — over-imagines on the unstable drone |
| **M2AC** | Model-based | Fixed length but **masks** the least-trustworthy imagined transitions | Masks high-uncertainty samples (filter, not length) | An *alternative* way to use uncertainty — masking vs adaptive truncation |
| **SAC** | Model-free | **No model rollouts** at all | — | The model-free reference — the baseline MACURA must beat on sample efficiency |

**What the comparison is meant to reveal:**
- **MACURA vs SAC** → does the model help? (sample-efficiency gap, in real steps).
- **MACURA vs MBPO** → does *adaptivity* matter, or is a fixed horizon fine? (stability on an
  unstable system — this is the headline contrast).
- **MACURA vs M2AC** → is *adapting the rollout length* better than *masking bad transitions*?
  (two uncertainty strategies, head-to-head).

## 7. Evaluation — the figures to produce

1. **Sample-efficiency curve** — evaluation return vs *real* environment steps, for all four
   algorithms, mean ± std over several seeds. Expected ordering by efficiency/stability:
   MACURA ≳ M2AC ≳ MBPO, all above SAC early on; MBPO expected to wobble or collapse late on the
   unstable task.
2. **Rollout-length / κ over training** (MACURA only) — the signature figure showing the rollout
   length adapting to model uncertainty.
3. **Failure rate while learning** — fraction of episodes that tumble / leave the envelope,
   over training — the safety/cost story (fewer real crashes to reach competence).
4. **Final-policy quality** — tracking error / hover stability of the best policy per algorithm.

## 8. Honest scope and risks

- This is a **faithful reproduction of the paper's mechanism in a new domain**, not a claim to
  beat the official numbers — the contribution is the *transfer* to an unstable system and the
  four-way comparison there.
- The drone simulator's API is version-sensitive; getting a stable, dense-reward task running is
  the first practical milestone.
- The dense-reward requirement is firm: MBRL (and therefore MACURA) is the wrong tool for
  sparse-reward tasks, and the project should not be framed around one.

## 9. Relationship to Project 2

Project 2 (baseline autonomous driving, model-free + future Neuro-Symbolic) is a **completely
separate** project. This drone project is where **MACURA itself is demonstrated**, on the kind
of dense-reward continuous-control task the method was designed for. Keeping the two apart means
each tells one clean story: *Project 1 = MACURA works (and generalises to an unstable system);
Project 2 = a simple, interpretable driving baseline that NeSy can later build on.*
