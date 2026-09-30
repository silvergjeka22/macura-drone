| | seeds | avg return while learning | drones broken after warm-up | final test return | success, % of a lap (test) | final test crash rate | return, last 10 evals |
|---|---|---|---|---|---|---|---|
| M2AC | 2 | **-9 (79 / -98)** | **25 (27 / 23)** | **287 (443 / 131)** | **27% (39% / 15%)** | **5% (0% / 10%)** | **210 (334 / 85)** |
| SAC | 2 | -476 (-495 / -458) | 157 (174 / 140) | -324 (-348 / -301) | 3% (2% / 3%) | 35% (33% / 37%) | -447 (-472 / -423) |

mean of the 2 seeds (each seed in brackets); a confidence interval needs 3 or more; bold = best. Test = the best checkpoint on 30 fresh scenarios. Success = share of a full lap flown per 10 s flight (autopilot on the same test flights: 54% at 1.5 m/s, 95% at 3 m/s).

| algorithm | seed | avg return while learning | drones broken after warm-up | final test return | success, % of a lap (test) | final test crash rate | return, last 10 evals | best step |
|---|---|---|---|---|---|---|---|---|
| M2AC | 6 | 79 | 27 | 443 | 39% | 0% | 334 | 48000 |
| M2AC | 7 | -98 | 23 | 131 | 15% | 10% | 85 | 45000 |
| SAC | 6 | -495 | 174 | -348 | 2% | 33% | -472 | 31000 |
| SAC | 7 | -458 | 140 | -301 | 3% | 37% | -423 | 29000 |

| M2AC - SAC | seed 6 | seed 7 | mean | M2AC better in |
|---|---|---|---|---|
| avg return while learning | +574 | +360 | +467 | 2 of 2 |
| drones broken after warm-up | -147 | -117 | -132 | 2 of 2 |
| final test return | +791 | +432 | +612 | 2 of 2 |
| lap progress (test) | +37% | +12% | +24% | 2 of 2 |
| final test crash rate | -33% | -27% | -30% | 2 of 2 |

| algorithm | seed | SAC updates per real step |
|---|---|---|
| M2AC | 6 | 12.0 |
| M2AC | 7 | 12.0 |
| SAC | 6 | 1 |
| SAC | 7 | 1 |

| algorithm | seed | fastest descent, 2nd half (m/s) | losing lift, 2nd half | test: losing lift |
|---|---|---|---|---|
| M2AC | 6 | 1.48 | 1.0% | 1.0% |
| M2AC | 7 | 1.15 | 1.3% | 1.3% |
| SAC | 6 | 1.93 | 2.4% | 1.2% |
| SAC | 7 | 1.67 | 1.4% | 0.9% |

M2AC kept the 25% least uncertain imagined steps; 1% of those were above MACURA's threshold