| algorithm | avg return while learning | drones broken | test return | test lap progress | test crash rate |
|---|---|---|---|---|---|
| MACURA | 69 | 28 | 333 | 31% | 4% |
| MBPO | -59 | 28 | 158 | 22% | 2% |
| M2AC | -28 | 26 | 267 | 24% | 6% |
| SAC | -475 | 154 | -312 | 7% | 35% |

| | seeds | avg return while learning | drones broken after warm-up | final test return | success, % of a lap (test) | final test crash rate | return, last 10 evals |
|---|---|---|---|---|---|---|---|
| MACURA | 4 | **69 [-62, 181]** | 28 [26, 34] | **295 [187, 449]** | **30% [17%, 40%]** | **2% [0%, 10%]** | **247 [138, 347]** |
| MBPO | 4 | -59 [-318, -6] | 28 [15, 205] | 133 [119, 305] | 20% [11%, 28%] | 3% [0%, 7%] | 121 [-530, 245] |
| M2AC | 4 | -28 [-98, 79] | **26 [23, 28]** | 196 [131, 443] | 20% [15%, 39%] | 5% [0%, 13%] | 184 [85, 334] |
| SAC | 4 | -475 [-495, -446] | 154 [139, 174] | -324 [-352, -221] | 7% [2%, 12%] | 35% [30%, 67%] | -437 [-472, -359] |

IQM over seeds [95% bootstrap CI]; bold = best. Test = the best checkpoint on 30 fresh scenarios. Success = share of a full lap flown per 10 s flight (autopilot on the same test flights: 54% at 1.5 m/s, 95% at 3 m/s).

| algorithm | seed | avg return while learning | drones broken after warm-up | final test return | success, % of a lap (test) | final test crash rate | return, last 10 evals | best step |
|---|---|---|---|---|---|---|---|---|
| MACURA | 4 | 181 | 31 | 363 | 40% | 10% | 317 | 26000 |
| MACURA | 5 | 52 | 34 | 227 | 20% | 0% | 178 | 45000 |
| MACURA | 6 | 85 | 26 | 449 | 40% | 3% | 347 | 48000 |
| MACURA | 7 | -62 | 26 | 187 | 17% | 0% | 138 | 46000 |
| MBPO | 4 | -47 | 28 | 119 | 19% | 3% | 139 | 44000 |
| MBPO | 5 | -318 | 205 | 129 | 21% | 7% | -530 | 18000 |
| MBPO | 6 | -6 | 28 | 305 | 28% | 3% | 245 | 49000 |
| MBPO | 7 | -71 | 15 | 137 | 11% | 0% | 104 | 34000 |
| M2AC | 4 | -45 | 26 | 170 | 21% | 13% | 187 | 49000 |
| M2AC | 5 | -11 | 28 | 222 | 19% | 0% | 180 | 48000 |
| M2AC | 6 | 79 | 27 | 443 | 39% | 0% | 334 | 48000 |
| M2AC | 7 | -98 | 23 | 131 | 15% | 10% | 85 | 45000 |
| SAC | 4 | -493 | 167 | -352 | 11% | 67% | -452 | 28000 |
| SAC | 5 | -446 | 139 | -221 | 12% | 30% | -359 | 41000 |
| SAC | 6 | -495 | 174 | -348 | 2% | 33% | -472 | 31000 |
| SAC | 7 | -458 | 140 | -301 | 3% | 37% | -423 | 29000 |

| | seeds | test return | lap progress per flight | crash rate | time losing lift | fastest descent (m/s) |
|---|---|---|---|---|---|---|
| MACURA | 4 | 333 [173, 422] | 31% [18%, 42%] | 4% [0%, 6%] | 0.3% [0.0%, 3.4%] | 0.79 [0.49, 1.25] |
| MBPO | 4 | 158 [99, 355] | 22% [14%, 30%] | 2% [0%, 14%] | 0.2% [0.0%, 1.4%] | 0.75 [0.37, 0.86] |
| M2AC | 4 | 267 [117, 383] | 24% [15%, 41%] | 6% [0%, 10%] | 0.9% [0.3%, 2.5%] | 0.89 [0.53, 1.37] |
| SAC | 4 | -312 [-385, -242] | 7% [4%, 10%] | 35% [28%, 68%] | 0.4% [0.3%, 0.7%] | 1.10 [1.01, 1.20] |

IQM over seeds [95% bootstrap CI]; each seed = its best checkpoint on the same 50 new scenarios (seeds 3000-3049).

| algorithm | seed | test return | lap progress per flight | crash rate | time losing lift | fastest descent (m/s) |
|---|---|---|---|---|---|---|
| MACURA | 4 | 422 | 42% | 4% | 0.4% | 0.94 |
| MACURA | 5 | 250 | 24% | 0% | 0.1% | 0.49 |
| MACURA | 6 | 417 | 38% | 6% | 3.4% | 1.25 |
| MACURA | 7 | 173 | 18% | 4% | 0.0% | 0.65 |
| MBPO | 4 | 183 | 21% | 2% | 1.4% | 0.81 |
| MBPO | 5 | 99 | 24% | 14% | 0.0% | 0.70 |
| MBPO | 6 | 355 | 30% | 0% | 0.4% | 0.37 |
| MBPO | 7 | 133 | 14% | 2% | 0.1% | 0.86 |
| M2AC | 4 | 280 | 26% | 6% | 0.4% | 1.05 |
| M2AC | 5 | 253 | 22% | 0% | 0.3% | 0.72 |
| M2AC | 6 | 383 | 41% | 6% | 1.5% | 1.37 |
| M2AC | 7 | 117 | 15% | 10% | 2.5% | 0.53 |
| SAC | 4 | -385 | 8% | 68% | 0.3% | 1.03 |
| SAC | 5 | -242 | 10% | 28% | 0.5% | 1.18 |
| SAC | 6 | -314 | 4% | 30% | 0.4% | 1.01 |
| SAC | 7 | -310 | 5% | 40% | 0.7% | 1.20 |

#### Test: MACURA vs MBPO

| MACURA - MBPO | seed 4 | seed 5 | seed 6 | seed 7 | mean | MACURA better in |
|---|---|---|---|---|---|---|
| test return | +238 | +151 | +62 | +40 | +123 | 4 of 4 |
| lap progress per flight | +21% | +0% | +8% | +4% | +8% | 4 of 4 |
| crash rate | +2% | -14% | +6% | +2% | -1% | 1 of 4 |
| time losing lift | -1.1% | +0.1% | +3.0% | -0.1% | +0.5% | 2 of 4 |

#### Test: MACURA vs M2AC

| MACURA - M2AC | seed 4 | seed 5 | seed 6 | seed 7 | mean | MACURA better in |
|---|---|---|---|---|---|---|
| test return | +142 | -4 | +34 | +56 | +57 | 3 of 4 |
| lap progress per flight | +16% | +2% | -3% | +3% | +4% | 3 of 4 |
| crash rate | -2% | +0% | +0% | -6% | -2% | 2 of 4 |
| time losing lift | -0.0% | -0.2% | +2.0% | -2.5% | -0.2% | 3 of 4 |

#### Test: MACURA vs SAC

| MACURA - SAC | seed 4 | seed 5 | seed 6 | seed 7 | mean | MACURA better in |
|---|---|---|---|---|---|---|
| test return | +807 | +492 | +731 | +483 | +628 | 4 of 4 |
| lap progress per flight | +34% | +14% | +34% | +13% | +24% | 4 of 4 |
| crash rate | -64% | -28% | -24% | -36% | -38% | 4 of 4 |
| time losing lift | +0.1% | -0.3% | +3.0% | -0.7% | +0.5% | 2 of 4 |

| algorithm | seed 4 | seed 5 | seed 6 | seed 7 |
|---|---|---|---|---|
| MACURA | macura_mbpo_seeds45 | macura_mbpo_seeds45 | macura_mbpo_seeds67 | macura_mbpo_seeds67 |
| MBPO | macura_mbpo_seeds45 | macura_mbpo_seeds45 | macura_mbpo_seeds67 | macura_mbpo_seeds67 |
| M2AC | m2ac_sac_seeds45 | m2ac_sac_seeds45 | m2ac_sac_seeds67 | m2ac_sac_seeds67 |
| SAC | m2ac_sac_seeds45 | m2ac_sac_seeds45 | m2ac_sac_seeds67 | m2ac_sac_seeds67 |