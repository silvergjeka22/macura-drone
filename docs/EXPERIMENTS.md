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
| `race3` | `race2` with a deadlier chute (up to 50% lift lost instead of 35%) | quick test, branch `race3` (section 6) |

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
| success: % of a lap per 10 s flight (test) | **27%** | 19% | 21% | 6% |
| drones broken after warm-up | 28 | **20** | 33 | 202 |

(Success replaces "a full lap within one flight", which was 0 for all four: a full lap in 10 s needs more than
2.4 m/s on average; the autopilot reaches 54% / 95% at 1.5 / 3 m/s on the same test flights.)
MACURA trusted 75% of a full 10-step imagination on average: 79% of imagined steps in fast descents, 96% in
normal flight. No policy reached the chute trap: the fastest descents stayed ≤ 1.4 m/s (lift loss starts at 1.2,
full at 2.2), so the lift loss rarely came into play. One seed shows a trend, not a result; seeds 1 and 2 give
the confidence intervals.

### Final study (decided 2026-09-27, before running it)
Seeds 1 and 2, 100,000 real steps, MACURA, MBPO and SAC, the environment unchanged; one Kaggle session per seed.
- **Longer, not a different task.** The one region where the models are wrong and disagree about it (the chute's
  lift loss) was barely reached at 50k (fastest descents 1.4 m/s) and MACURA's best checkpoint was its last
  evaluation. Training longer is the fair way to reach it; a new environment would need a screening run first.
- **Budget.** Estimated from the seed-0 Kaggle timings (model fit grows linearly with the data), per seed at 100k:
  MACURA ~4.5 h, MBPO ~3.4 h, M2AC ~3.4 h, SAC ~0.5 h. All four: ~11.7 h + setup, too close to Kaggle's 12 h limit,
  and ~25 h for two seeds (budget 23 h). Without M2AC: ~9 h per seed, ~18 h in total. MBPO is the method MACURA
  builds on, so MACURA vs MBPO stays the main comparison; M2AC appears in the 50k pilot only.
- **Safety.** Training stops at 11 h into a session (the run is cut at its next evaluation and saved; the log
  records `stopped_early_at`); a killed session would save nothing.
- **Two seeds** give no confidence interval: the tables show both seeds' values; a method is only called better if
  both seeds agree.
- Seed 0 (50k) is analysed on its own; the analysis compares runs of the same length only.

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

## 6. RACE3: a deadlier chute (branch `race3`, decided 2026-09-27)

`MACURA_TASK=race3` = `race2` with one change, `vrs_loss` 0.35 -> 0.50. Where the lift loss starts (1.2 m/s) and
everything else (course, reward, crash cost, wind, package, stabiliser, protocol) is unchanged; the other tasks'
settings are identical to before (checked). Real and imagined reward / crash flags: 0 mismatches (4,396 steps, 405
losing lift).

Why (*measured*, hand-written autopilot at 3 m/s, 30 scenarios 5000-5029, discounted return gamma 0.99):

| chute speed | race2 (35%) | race3 (50%) |
|---|---|---|
| safe, 1.0 m/s | 0% crash, 205 | 0% crash, 205 |
| moderate dive, 1.5 m/s | 7% crash, **213** | 57% crash, 168 |
| dive, 2.0 m/s | 27% crash, 202 | 77% crash, 149 |
| random warm-up steps losing lift | 1.0% | 0.9% |

In race2 a moderate dive paid slightly MORE than flying the chute safely, so avoiding the trap could not help MACURA;
in race3 it clearly loses, while random warm-up still almost never loses lift (the trap stays new to the world model).
Rejected: 65% (the safe line crashes 23%); starting the lift loss at 0.8 m/s instead of 1.2 (random warm-up would lose
lift in 4.2% of steps, so the trap would no longer be new). Seed-0 race2 pilot policies in the chute (test scenarios):
MACURA sinks at up to 1.06 m/s in 90% of its steps (max 1.82), MBPO 0.70 (max 1.61): MACURA flies it faster, so a
deadlier trap can also hurt MACURA (as in the delivery pilot).

**Quick test** (Kaggle, seed 0, 10,000 steps = 5,000 warm-up + 5,000 learning, MACURA / MBPO / SAC, *measured*):
- The race2 half reproduced the first 10,000 steps of the seed-0 race2 pilot **exactly** (every evaluation identical):
  runs are deterministic on Kaggle for a given seed, and the race3 branch leaves race2 unchanged.
- MACURA - MBPO, race3 vs race2: avg return while learning -9 vs +7, drones broken +6 vs +9, final test return +187 vs
  -131, test crash -17% vs +30%. The signs flip, so this is noise: race3 already diverges from race2 during the random
  warm-up (34 vs 32 drones broken at step 5,000), and after 5,000 learning steps the drones barely fly.
- The chute: lift loss below 1% of the evaluation time, 0% in MACURA's test; the only fast descent (2.2 m/s) was a
  beginner policy that crashed in every flight. The notebook's check counted it; it now looks only at the second half
  of training and the final test.
- One sign of the mechanism: MACURA trusted fast descents less in race3 (58% of the steps) than in race2 (81%), lower
  at all 5 evaluations; average imagined trip 5.0 vs 6.1 of 10.

**Final race3 test** (decided 2026-09-27, before running it):
- race3, **seeds 0-3**, **50,000 steps**, MACURA, MBPO and SAC; seeds 0, 1 on one Kaggle account and 2, 3 on another,
  at the same time; in each session MACURA and MBPO of both seeds train in parallel (one process each), SAC after.
- **MBPO: 12 SAC updates per real step** (was 8) = MACURA's average in the race2 pilot (11.7), so neither method
  simply trains more (MACURA keeps its adaptive updates, up to 16). Exploration unchanged (paper protocol, disclosed).
- **Decision rule**: race3 becomes the final study if MACURA's average return while learning is higher than MBPO's
  in **at least 3 of the 4 seeds** and on their average; otherwise the race2 plan stays. Drones broken, test return,
  crash rate, updates per step and the chute check are reported either way, and the race2 pilot is reported next to
  the race3 runs whichever task is chosen.
