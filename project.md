# Project — MACURA learns a Backflip

**Teaching Pogo (a planar one-legged pogo-stick gymnast) to backflip, and using
that aggressive, unstable maneuver to compare the four algorithms of the MACURA
paper (MACURA, MBPO, M2AC, SAC).**

> Paper: *Trust the Model Where It Trusts Itself — Model-Based Actor-Critic with
> Uncertainty-Aware Rollout Adaption*, Frauenknecht, Eisele, Subhasish, Solowjow,
> Trimpe, ICML 2024 (arXiv:2405.19014).

This document is the **project idea only** — the task, the goal, and why a backflip
fits the paper. No code.

## 1. The idea in one paragraph

Model-based RL (MBRL) is sample-efficient because the policy trains on *imagined*
rollouts from a learned dynamics model. The danger is **model exploitation**: imagine
too far and the model's small errors compound, so the policy exploits fantasy
dynamics. MACURA's contribution is to make the rollout length **adaptive to the
model's own uncertainty** — "trust the model where it trusts itself." This project
takes that mechanism to a task *not* in the paper: a **backflip**, where the model
is most uncertain exactly at the aggressive airborne/landing phase.

## 2. Why a backflip is the right test for MACURA

- **Dynamics are smooth and learnable** between contacts, so the ensemble can fit
  them and MBRL is viable.
- **The maneuver is aggressive and unstable** → a small model error at takeoff or
  landing compounds fast. This is precisely the regime where a *fixed* rollout
  horizon over-imagines the flight and the policy degrades (a face-plant), and
  where MACURA's *adaptive* truncation should shine.
- **Non-paper** → the paper validates on MuJoCo locomotion (Hopper, Walker2d,
  HalfCheetah, Ant, Humanoid). A backflip on a custom body is a genuine transfer.

## 3. The task

- **Body:** Pogo — a planar one-legged pogo-stick with a face (custom MuJoCo model,
  `src/envs/assets/pogo.xml`). Motion in the x–z plane; the torso hinge is unlimited so
  it can rotate a full 360°.
- **Observation:** torso height, torso angle (as sin/cos), the three leg-joint
  angles, foot clearance, and all six velocities (13-dim).
- **Action:** continuous torque on hip / knee / ankle (3-dim).
- **Reward:** dense and shaped — jump (foot clearance), spin while airborne (the
  flip), land upright while grounded, minus control effort. Analytic in `(obs,
  action)` so imagined and real transitions are scored identically. `w_rotation` is
  a curriculum knob.
- **Failure:** the torso collapsing to the mat (a face-plant) ends the episode.
- **Success:** a full 360° rotation landed upright.

## 4. The goal

1. **Reproduce the paper's mechanism faithfully** on the backflip (the four
   algorithms, shared SAC backbone and ensemble, only the rollout differs).
2. **Show MACURA's two claimed advantages** here: sample efficiency (reach a
   competent flip in fewer real steps than SAC) and stability (fewer face-plants
   than fixed-horizon MBPO, which over-imagines the flight).
3. **Demonstrate the adaptive mechanism directly** — rollout length short when the
   model is uncertain (the flip), growing as it becomes reliable; κ falling.

## 5. The four-algorithm comparison

| Algorithm | Rollout strategy | Uncertainty use | Role |
|---|---|---|---|
| **MACURA** | uncertainty-adaptive truncation (κ vs GJS) | drives κ | the proposed method — best efficiency *and* stability |
| **MBPO** | fixed truncated-linear schedule | none | over-imagines the flight on an unstable maneuver |
| **M2AC** | fixed length + mask worst transitions | filter, not length | masking vs truncation |
| **SAC** | no model rollouts | — | model-free reference to beat on sample efficiency |

## 6. Figures to produce

1. Sample-efficiency curve (return vs real steps, mean ± std over seeds).
2. **Stuck-backflip (success) rate** and **face-plant rate** over training — the
   headline stability story.
3. Final policy quality per algorithm.
4. MACURA signature — rollout length & κ over training.
5. A **live flight demo** — MACURA sticking the flip vs MBPO face-planting.

## 7. Honest scope

A backflip is hard to discover from scratch, so the config supports a **curriculum**
(jump → half-flip → full flip). This is a faithful reproduction of the paper's
mechanism on a new unstable task, **not a guaranteed win**. If MACURA does not win
after a fair run, that is reported and diagnosed — numbers are never doctored.
