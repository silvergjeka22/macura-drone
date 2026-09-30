| | seeds | avg return while learning | drones broken after warm-up | final test return | success, % of a lap (test) | final test crash rate | return, last 10 evals |
|---|---|---|---|---|---|---|---|
| MACURA | 2 | **12 (85 / -62)** | 26 (26 / 26) | **318 (449 / 187)** | **28% (40% / 17%)** | 2% (3% / 0%) | **242 (347 / 138)** |
| MBPO | 2 | -39 (-6 / -71) | **22 (28 / 15)** | 221 (305 / 137) | 20% (28% / 11%) | 2% (3% / 0%) | 174 (245 / 104) |

mean of the 2 seeds (each seed in brackets); a confidence interval needs 3 or more; bold = best. Test = the best checkpoint on 30 fresh scenarios. Success = share of a full lap flown per 10 s flight (autopilot on the same test flights: 54% at 1.5 m/s, 95% at 3 m/s).

| algorithm | seed | avg return while learning | drones broken after warm-up | final test return | success, % of a lap (test) | final test crash rate | return, last 10 evals | best step |
|---|---|---|---|---|---|---|---|---|
| MACURA | 6 | 85 | 26 | 449 | 40% | 3% | 347 | 48000 |
| MACURA | 7 | -62 | 26 | 187 | 17% | 0% | 138 | 46000 |
| MBPO | 6 | -6 | 28 | 305 | 28% | 3% | 245 | 49000 |
| MBPO | 7 | -71 | 15 | 137 | 11% | 0% | 104 | 34000 |

| MACURA - MBPO | seed 6 | seed 7 | mean | MACURA better in |
|---|---|---|---|---|
| avg return while learning | +91 | +9 | +50 | 2 of 2 |
| drones broken after warm-up | -2 | +11 | +4 | 1 of 2 |
| final test return | +145 | +51 | +98 | 2 of 2 |
| lap progress (test) | +12% | +5% | +9% | 2 of 2 |
| final test crash rate | +0% | +0% | +0% | 0 of 2 |

| algorithm | seed | SAC updates per real step |
|---|---|---|
| MACURA | 6 | 10.8 |
| MACURA | 7 | 11.5 |
| MBPO | 6 | 12.0 |
| MBPO | 7 | 12.0 |

| algorithm | seed | fastest descent, 2nd half (m/s) | losing lift, 2nd half | test: losing lift |
|---|---|---|---|---|
| MACURA | 6 | 1.49 | 2.5% | 2.1% |
| MACURA | 7 | 0.87 | 0.2% | 0.0% |
| MBPO | 6 | 1.07 | 0.8% | 0.4% |
| MBPO | 7 | 0.86 | 0.3% | 0.2% |

MACURA trusted 71% of a full 10-step imagination (average trip 7.1/10); steps passing its trust check: fast descents 74%, normal flight 94%
MBPO trusted 100% of what it imagined (no trust check); 13% of it was above MACURA's threshold