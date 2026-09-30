| | seeds | avg return while learning | drones broken after warm-up | final test return | success, % of a lap (test) | final test crash rate | return, last 10 evals |
|---|---|---|---|---|---|---|---|
| M2AC | 2 | **-28 (-45 / -11)** | **27 (26 / 28)** | **196 (170 / 222)** | **20% (21% / 19%)** | **7% (13% / 0%)** | **184 (187 / 180)** |
| SAC | 2 | -470 (-493 / -446) | 153 (167 / 139) | -287 (-352 / -221) | 11% (11% / 12%) | 48% (67% / 30%) | -405 (-452 / -359) |

mean of the 2 seeds (each seed in brackets); a confidence interval needs 3 or more; bold = best. Test = the best checkpoint on 30 fresh scenarios. Success = share of a full lap flown per 10 s flight (autopilot on the same test flights: 54% at 1.5 m/s, 95% at 3 m/s).

| algorithm | seed | avg return while learning | drones broken after warm-up | final test return | success, % of a lap (test) | final test crash rate | return, last 10 evals | best step |
|---|---|---|---|---|---|---|---|---|
| M2AC | 4 | -45 | 26 | 170 | 21% | 13% | 187 | 49000 |
| M2AC | 5 | -11 | 28 | 222 | 19% | 0% | 180 | 48000 |
| SAC | 4 | -493 | 167 | -352 | 11% | 67% | -452 | 28000 |
| SAC | 5 | -446 | 139 | -221 | 12% | 30% | -359 | 41000 |

| M2AC - SAC | seed 4 | seed 5 | mean | M2AC better in |
|---|---|---|---|---|
| avg return while learning | +448 | +435 | +442 | 2 of 2 |
| drones broken after warm-up | -141 | -111 | -126 | 2 of 2 |
| final test return | +522 | +444 | +483 | 2 of 2 |
| lap progress (test) | +9% | +7% | +8% | 2 of 2 |
| final test crash rate | -53% | -30% | -42% | 2 of 2 |

| algorithm | seed | SAC updates per real step |
|---|---|---|
| M2AC | 4 | 12.0 |
| M2AC | 5 | 12.0 |
| SAC | 4 | 1 |
| SAC | 5 | 1 |

| algorithm | seed | fastest descent, 2nd half (m/s) | losing lift, 2nd half | test: losing lift |
|---|---|---|---|---|
| M2AC | 4 | 1.43 | 0.7% | 0.1% |
| M2AC | 5 | 1.10 | 0.8% | 0.9% |
| SAC | 4 | 1.58 | 1.5% | 0.1% |
| SAC | 5 | 1.63 | 1.2% | 0.5% |

M2AC kept the 25% least uncertain imagined steps; 3% of those were above MACURA's threshold