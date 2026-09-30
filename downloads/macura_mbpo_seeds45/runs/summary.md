| | seeds | avg return while learning | drones broken after warm-up | final test return | success, % of a lap (test) | final test crash rate | return, last 10 evals |
|---|---|---|---|---|---|---|---|
| MACURA | 2 | **116 (181 / 52)** | **32 (31 / 34)** | **295 (363 / 227)** | **30% (40% / 20%)** | 5% (10% / 0%) | **247 (317 / 178)** |
| MBPO | 2 | -182 (-47 / -318) | 116 (28 / 205) | 124 (119 / 129) | 20% (19% / 21%) | 5% (3% / 7%) | -196 (139 / -530) |

mean of the 2 seeds (each seed in brackets); a confidence interval needs 3 or more; bold = best. Test = the best checkpoint on 30 fresh scenarios. Success = share of a full lap flown per 10 s flight (autopilot on the same test flights: 54% at 1.5 m/s, 95% at 3 m/s).

| algorithm | seed | avg return while learning | drones broken after warm-up | final test return | success, % of a lap (test) | final test crash rate | return, last 10 evals | best step |
|---|---|---|---|---|---|---|---|---|
| MACURA | 4 | 181 | 31 | 363 | 40% | 10% | 317 | 26000 |
| MACURA | 5 | 52 | 34 | 227 | 20% | 0% | 178 | 45000 |
| MBPO | 4 | -47 | 28 | 119 | 19% | 3% | 139 | 44000 |
| MBPO | 5 | -318 | 205 | 129 | 21% | 7% | -530 | 18000 |

| MACURA - MBPO | seed 4 | seed 5 | mean | MACURA better in |
|---|---|---|---|---|
| avg return while learning | +227 | +370 | +299 | 2 of 2 |
| drones broken after warm-up | +3 | -171 | -84 | 1 of 2 |
| final test return | +244 | +98 | +171 | 2 of 2 |
| lap progress (test) | +21% | -1% | +10% | 1 of 2 |
| final test crash rate | +7% | -7% | +0% | 1 of 2 |

| algorithm | seed | SAC updates per real step |
|---|---|---|
| MACURA | 4 | 9.8 |
| MACURA | 5 | 11.5 |
| MBPO | 4 | 12.0 |
| MBPO | 5 | 12.0 |

| algorithm | seed | fastest descent, 2nd half (m/s) | losing lift, 2nd half | test: losing lift |
|---|---|---|---|---|
| MACURA | 4 | 1.20 | 0.5% | 0.2% |
| MACURA | 5 | 1.35 | 1.7% | 0.2% |
| MBPO | 4 | 1.27 | 0.6% | 1.0% |
| MBPO | 5 | 1.56 | 1.2% | 0.0% |

MACURA trusted 67% of a full 10-step imagination (average trip 6.7/10); steps passing its trust check: fast descents 60%, normal flight 94%
MBPO trusted 100% of what it imagined (no trust check); 13% of it was above MACURA's threshold