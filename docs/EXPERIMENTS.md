# Experiment log: why every setting has its value

Rules followed throughout: every environment change applies to all four algorithms and is decided before
the run it is used in; nothing is tuned on the evaluation scenarios (selection 100-119, final test 1000-1029);
other tasks stay bit-identical (new features sit behind config flags that are off by default); reward and
termination are pure functions of (observation, action), so real and imagined transitions are scored alike.
Numbers marked *measured* come from runs; anything else is labelled *not measured*.

## 1. Task history

| task (`MACURA_TASK`) | what it is | outcome |
|---|---|---|
| `""` cage | land inside an open-top cage under wind | learnable, but the world model was not more uncertain where the landing is decided, so it could not separate MACURA from MBPO |
| `delivery` | land a package of random weight; lift loss in fast descents | the lift loss was not new to the models: early free-falling flights already sank fast (disagreement only 1.5x there) |
| `delivery2` | powered lift loss (upright + sinking along the rotor axis + pushing), higher start | policies hovered: the discounted value of landing was too close to hovering |
| `race` | laps around a 3-D course with a chute; open-ended reward (speed along the course) | 10k-step test: nobody learned to fly (laps ~0, 55-100% crashes) |
| **`race2`** | `race` + altitude-hold stabiliser, racing-line sensor, 2.5 m off-course limit | **the study** |

Landing-zone physics tried offline and rejected (removed from the code in the refactor; they were off in every
task): turbulence near the pad, ground effect, a deterministic wake, hidden action noise, a solid touchdown
platform. Each made the ensemble *agree more* near the pad (GJS near/transit 0.33-0.87x), so MACURA would have
trusted the hard region more, not less.

## 2. RACE2 design (decided 2026-09-25, before its first Kaggle run)

Course: a 6 x 6 m rounded square, clockwise, P1 2 m → P2 4 m → chute (near-vertical 3 m drop) → P3 1 m →
P4 1.5 m, 24.3 m per lap, two pillars; random start point; 10 s flights; package 0-0.2 kg and steady wind
up to 0.5 N (both observed); hidden OU gusts (0.4 N std, correlation 0.95) and 4% rotor-thrust noise.
Reward per step: speed along the course near it, minus metres outside a 0.35 m tube, + 0.1 upright, small
spin and effort terms; crash (flip, ground < 0.12 m, pillar, > 2.5 m off course) costs 500.

Why the race needed RACE2 (*measured*, 60 flights each, before any training):

| a beginner that ... | race: survives | race2: survives |
|---|---|---|
| does nothing (action 0) | 1.1 s | 4.3 s |
| acts at random | 0.3 s (flips) | 3.1 s |

- **Altitude-hold stabiliser** (`ctrl_mode="althold"`): actions = sideways acceleration x, y (tilt ≤ 40°),
  climb rate (±3 m/s), yaw rate; action 0 = level hover for the current mass. Same rotors, limits and lift-loss
  physics. A climb-rate stick was chosen over a vertical-acceleration stick because random vertical accelerations
  integrate into fast sinks: beginners lost lift in 9% of random steps (the model learned the trap before any
  policy tried the chute); with the climb-rate stick 0.3%.
- **Off-course limit 2.5 m** (was 1.5): a level hover drifts off in the gusts within ~3 s.
- **Racing-line sensor** (+6 observation dims): vector to the nearest course point and the direction 0.5 m ahead,
  a function of the position already observed.
- Screening (MBPO, 10k steps, 1 seed, screening scenarios): althold alone 70% crashes; + 2.5 m 30%; + sensor 0%;
  return -451 → -199 → +96 (*measured*).

Hand-written autopilot, same stabiliser, 30 scenarios (seeds 5000-5029, *measured*); discounted = gamma 0.99:

| autopilot | laps | crash | raw return | discounted, crash cost 0 / 100 / 250 / 500 |
|---|---|---|---|---|
| careful 1.5 m/s (chute 0.8) | 0.52 | 3% | 635 | 118 / 117 / 116 / 114 |
| **fast 3 m/s (chute 1.1)** | **0.93** | **0%** | **1180** | **209 / 209 / 209 / 209** |
| dive 3 m/s (chute 2.0) | 0.90 | 27% | 977 | 216 / 214 / 209 / 202 |
| dive 3 m/s (chute 3.0) | 0.69 | 70% | 446 | 199 / 189 / 174 / 150 |
| hover (action 0) | -0.02 | 97% | -601 | -21 / -40 / -67 / -114 |

Crash cost 500: at 250 a moderate chute dive already pays as much as flying safely. On the selection scenarios
(100-119) the autopilot scores 673 (1.5 m/s) and 1172 (3 m/s); these are the reference lines in the learning curves.

Is the chute new to the model? Offline check (7-member ensemble trained on random + pink-noise beginners and slow
careful flights, scored on new flights, ratio vs held-out slow flights, *measured*): chute dives losing lift:
disagreement 6.1x (median), error 5.3x (mean); fast flat racing: 1.2x / 0.9x. The chute is the one place where
the model is confidently wrong.

## 3. Protocol (the MACURA paper, App. D; M2AC as in Pan et al. 2020)

| setting | value | why |
|---|---|---|
| ensemble | 7 members, 5 elites, 20% holdout (≤ 5000), ≤ 10 epochs, patience 3 | paper / mbrl-lib; the epoch cap keeps a seed of four algorithms < 12 h |
| model rounds | every 250 real steps, 25,000 imagined trips from real states | paper uses ~400 trips per real step; 100 keeps the run inside Kaggle's 12 h |
| SAC batch | 5% real, 95% imagined; batch 256 | paper |
| imagined-data lifetime | 4 model rounds (1000 real steps), all three model-based methods | MACURA reference code (`ReplayBufferDynamicLifeTime`) |
| MACURA | t_max 10, ζ 0.95, **ξ 0.5**, Gmax 16, Eq. 22 on the live window | see below |
| MBPO | horizon 1 → 10 over steps 5000-7500, 8 updates per step | paper; ramp starts after the warm-up |
| M2AC | paper mode: keep (H-h)/(2(H+1)) per step (25% overall), α = 1e-3, non-stop, 8 updates | Pan et al. 2020 |
| SAC | 1 update per real step | standard |
| warm-up | 5000 random steps, all four | MACURA reference code (`initial_exploration_steps`) |
| exploration | MACURA pink noise; MBPO, M2AC deterministic; SAC samples its policy | as in the paper; **a known confound** (paper Fig. 5); `MACURA_EXPLORATION=equal` removes it |
| evaluation | 20 fixed scenarios every 1000 steps; best checkpoint → 30 fresh scenarios | selection and test never share scenarios |

**Warm-up 5000 and ξ 0.5** (decided 2026-09-26, before the seed-0 run used in the study). The first race2 seed-0
run used 500 warm-up steps and ξ 1: early world models spiked in disagreement (95% quantile of first-step GJS
110-442 around steps 1500-2000, vs ~6-14 later), κ as a running mean over all rounds stayed inflated, and MACURA's
filter trusted almost everything (imagined trips ~9.7 of 10), so that run did not test MACURA's idea. Local check
with 5000 warm-up steps (screening scenarios, 1 seed), mean imagined trip at 5k / 6k / 7k / 8k steps (*measured*):
ξ 1: 7.8 / 7.0 / 9.2 (filter switches off again); ξ 0.5: 4.3 / 4.7 / 6.7 / 6.5; ξ 0.25: 0.4 / 0.5 / 0.7 / 2.0.
ξ 0.5 was chosen from this mechanism check, not from returns (paper App. D.2: ξ is the one per-task knob).

**Imagined-data lifetime and Eq. 22.** With one FIFO buffer that is always full, Eq. 22 always gave MACURA its
maximum 16 updates. With the reference code's lifetime rule, the live window's fill level follows the trip length,
so MACURA gets fewer updates when it trusts its model less (paper App. D.5).

## 4. Runs

### RACE2 seed 0, 50,000 steps, final protocol (Kaggle, *measured*, one seed)

| | MACURA | MBPO | M2AC | SAC |
|---|---|---|---|---|
| avg return while learning | **7** | -79 | -74 | -513 |
| return, last 10 evals | **196** | 141 | 102 | -475 |
| final test return (30 fresh scenarios) | **284** | 111 | 163 | -423 |
| final test crash rate | **3%** | 13% | **3%** | 63% |
| laps per 10 s flight (test) | **0.27** | 0.19 | 0.21 | 0.06 |
| drones broken after warm-up | 28 | **20** | 33 | 202 |

MACURA trusted 75% of a full 10-step imagination on average: 79% of imagined steps in fast descents, 96% in
normal flight. No policy reached the chute trap: the fastest descents stayed ≤ 1.4 m/s (lift loss starts at 1.2,
full at 2.2), so the lift loss rarely came into play. One seed shows a trend, not a result; seeds 1 and 2 give
the confidence intervals.

### Not run (paper results not reproduced here)
Exploration ablation (paper Fig. 5), ξ sweep (Fig. 6), component ablations (Fig. 7), SAC to its asymptote, other
MuJoCo tasks. The optional control run `MACURA_ALGOS=mbpo MBPO_UTD=16` (MBPO with MACURA's maximum update budget)
would test whether "more updates" alone explains a MACURA lead (*not measured*).

## 5. Refactor check (2026-09-27)

The code was reorganised (dead physics removed from the environment, `plots.py` split into `analysis/results.py`,
`viz/figures.py` and `viz/video.py`, a simulator added). Verified against the code that ran seed 0 (commit aa0f744):
environment trajectories, rewards, terminations, random-number state and MJCF identical for all five tasks, and
5,600-step race2 training logs of all four algorithms byte-identical. Seed 0 therefore stays valid next to seeds
1 and 2 run with the new code.
