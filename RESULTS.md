# Results

All numbers below are produced by `python run_all.py` and read from `results/summary/summary.json`. Learned agents are trained with 10 seeds for 20,000 steps and evaluated exactly on the full state space of the simulator, so evaluation adds no sampling noise. Values for learned agents are means with 95% bootstrap confidence intervals over seeds. Rules and DP policies are deterministic and evaluated exactly. SLA cost is the expected class-weighted miss penalty per task, 0.5 (10 v_crit + 2 v_ana).

## Main comparison

| Policy | v_crit (%) | v_ana (%) | All (%) | SLA cost | Lat. crit (ms) | Lat. ana (ms) | Edge (%) |
|---|---|---|---|---|---|---|---|
| RR | 36.7 | 33.3 | 35.0 | 2.170 | 59.9 | 59.9 | 66.7 |
| JSQ | 51.8 | 14.4 | 33.1 | 2.735 | 61.5 | 61.5 | 100.0 |
| SLA rule | 11.7 | 0.0 | 5.8 | 0.583 | 45.4 | 39.7 | 94.2 |
| Reservation rule (tuned) | 1.3 | 24.1 | 12.7 | 0.304 | 34.4 | 49.3 | 87.3 |
| A2C flat | 48.1 [41.0, 51.8] | 12.6 [9.7, 14.2] | 30.3 [25.4, 33.0] | 2.530 [2.147, 2.731] | 59.9 [57.3, 61.4] | 60.1 [57.4, 61.5] | 99.3 [97.9, 100.0] |
| A2C weighted | 49.3 [45.5, 52.1] | 10.9 [7.6, 13.5] | 30.1 [26.5, 32.8] | 2.572 [2.350, 2.740] | 58.8 [55.6, 61.3] | 58.6 [55.0, 61.2] | 99.2 [98.3, 100.0] |
| DQN flat | 21.5 [19.6, 23.1] | 2.3 [2.0, 2.7] | 11.9 [11.0, 12.7] | 1.100 [1.005, 1.178] | 46.0 [45.5, 46.6] | 44.9 [43.4, 47.5] | 95.2 [95.0, 95.5] |
| DQN weighted | 6.4 [5.4, 7.5] | 15.5 [14.6, 16.6] | 11.0 [10.6, 11.4] | 0.476 [0.433, 0.526] | 37.4 [36.8, 38.1] | 46.9 [45.9, 48.5] | 91.9 [91.5, 92.3] |
| PPO flat | 13.6 [11.7, 16.1] | 0.8 [0.6, 1.0] | 7.2 [6.3, 8.3] | 0.689 [0.596, 0.808] | 45.4 [45.0, 45.9] | 40.6 [40.0, 41.3] | 94.3 [94.0, 94.6] |
| PPO weighted | 7.5 [6.4, 8.6] | 12.8 [11.6, 14.1] | 10.2 [9.7, 10.7] | 0.506 [0.457, 0.551] | 39.0 [38.4, 39.6] | 45.1 [44.6, 45.6] | 92.4 [91.9, 92.8] |
| Optimal for flat reward (DP) | 10.2 | 0.0 | 5.1 | 0.510 | 44.3 | 46.6 | 94.9 |
| Optimal for weighted reward (DP) | 2.4 | 14.3 | 8.3 | 0.264 | 37.4 | 46.4 | 91.7 |
| Optimal SLA cost (DP) | 2.1 | 15.8 | 8.9 | 0.261 | 36.9 | 46.6 | 91.1 |

The tuned reservation rule sends analytics tasks to the edge only when the shorter queue holds at most 0 tasks.

## Distance to the optimum

| Policy | SLA cost | Above optimum |
|---|---|---|
| RR | 2.170 | 730% |
| JSQ | 2.735 | 946% |
| SLA rule | 0.583 | 123% |
| Reservation rule (tuned) | 0.304 | 16% |
| A2C flat | 2.530 | 868% |
| A2C weighted | 2.572 | 884% |
| DQN flat | 1.100 | 321% |
| DQN weighted | 0.476 | 82% |
| PPO flat | 0.689 | 164% |
| PPO weighted | 0.506 | 93% |
| Optimal for flat reward (DP) | 0.510 | 95% |
| Optimal for weighted reward (DP) | 0.264 | 1% |
| Optimal SLA cost (DP) | 0.261 | 0% |

## Weighted against flat reward

Difference is weighted minus flat (percentage points for rates). A12 is the Vargha-Delaney effect size, the probability that a weighted seed has a higher value than a flat seed. p values are two-sided Mann-Whitney U tests, exact when the samples have no ties and normal-approximated otherwise (only the A2C rows have ties), Holm-corrected over all rows. The last column counts seeds where the weighted agent is lower than the flat agent trained with the same seed.

| Algo | Metric | Diff | A12 | p | p (Holm) | W lower |
|---|---|---|---|---|---|---|
| PPO | critical misses | -6.1 | 0.00 | <0.001 | <0.001 | 10/10 |
| PPO | analytics misses | 12.1 | 1.00 | <0.001 | <0.001 | 0/10 |
| PPO | SLA cost | -0.183 | 0.08 | <0.001 | 0.005 | 10/10 |
| PPO | all misses | 3.0 | 0.90 | 0.002 | 0.009 | 1/10 |
| DQN | critical misses | -15.1 | 0.00 | <0.001 | <0.001 | 10/10 |
| DQN | analytics misses | 13.2 | 1.00 | <0.001 | <0.001 | 0/10 |
| DQN | SLA cost | -0.624 | 0.00 | <0.001 | <0.001 | 10/10 |
| DQN | all misses | -0.9 | 0.19 | 0.019 | 0.093 | 8/10 |
| A2C | critical misses | 1.2 | 0.40 | 0.444 | 1.000 | 6/10 |
| A2C | analytics misses | -1.7 | 0.29 | 0.120 | 0.479 | 7/10 |
| A2C | SLA cost | 0.042 | 0.41 | 0.540 | 1.000 | 5/10 |
| A2C | all misses | -0.3 | 0.39 | 0.424 | 1.000 | 6/10 |

## Weighted agents against rules

Seeds better counts seeds where the agent has a lower value than the rule. p is a two-sided sign test.

| Algo | Rule | Metric | Agent mean | Rule | Seeds better | p |
|---|---|---|---|---|---|---|
| PPO | threshold | critical misses | 7.5 | 11.7 | 10/10 | 0.002 |
| PPO | threshold | SLA cost | 0.506 | 0.583 | 8/10 | 0.109 |
| PPO | reservation | critical misses | 7.5 | 1.3 | 0/10 | 0.002 |
| PPO | reservation | SLA cost | 0.506 | 0.304 | 0/10 | 0.002 |
| DQN | threshold | critical misses | 6.4 | 11.7 | 10/10 | 0.002 |
| DQN | threshold | SLA cost | 0.476 | 0.583 | 9/10 | 0.021 |
| DQN | reservation | critical misses | 6.4 | 1.3 | 0/10 | 0.002 |
| DQN | reservation | SLA cost | 0.476 | 0.304 | 0/10 | 0.002 |
| A2C | threshold | critical misses | 49.3 | 11.7 | 0/10 | 0.002 |
| A2C | threshold | SLA cost | 2.572 | 0.583 | 0/10 | 0.002 |
| A2C | reservation | critical misses | 49.3 | 1.3 | 0/10 | 0.002 |
| A2C | reservation | SLA cost | 2.572 | 0.304 | 0/10 | 0.002 |

## Penalty ratio (PPO, mean penalty 6)

| w_crit:w_ana | v_crit (%) | v_ana (%) | SLA cost | DP v_crit (%) | DP v_ana (%) | DP SLA cost |
|---|---|---|---|---|---|---|
| 6:6 | 13.6 [11.7, 16.1] | 0.8 [0.6, 1.0] | 0.689 [0.596, 0.808] | 10.2 | 0.0 | 0.510 |
| 8:4 | 13.7 [11.1, 16.6] | 4.9 [4.2, 5.5] | 0.734 [0.611, 0.875] | 4.8 | 7.9 | 0.319 |
| 9:3 | 10.8 [8.8, 13.1] | 7.9 [7.3, 8.5] | 0.617 [0.524, 0.728] | 3.9 | 9.9 | 0.292 |
| 10:2 | 7.5 [6.4, 8.6] | 12.8 [11.6, 14.1] | 0.506 [0.457, 0.551] | 2.4 | 14.3 | 0.264 |
| 11:1 | 5.1 [4.2, 5.9] | 19.5 [18.3, 20.9] | 0.448 [0.415, 0.482] | 1.7 | 17.7 | 0.262 |

## Edge load (PPO)

Load rho is the offered edge load, arrival rate over total drain rate of both edge nodes.

| Drain | rho | Flat v_crit (%) | Weighted v_crit (%) | Flat cost | Weighted cost | p cost | DP flat cost | DP weighted cost | Optimum | Reservation |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.9 | 0.67 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.940 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.8 | 0.75 | 0.1 [0.1, 0.1] | 0.1 [0.1, 0.1] | 0.006 [0.006, 0.006] | 0.006 [0.006, 0.007] | 0.853 | 0.003 | 0.003 | 0.003 | 0.004 |
| 0.7 | 0.86 | 3.4 [3.2, 3.6] | 2.9 [2.6, 3.2] | 0.172 [0.161, 0.182] | 0.150 [0.136, 0.163] | 0.035 | 0.057 | 0.048 | 0.047 | 0.061 |
| 0.6 | 1.00 | 13.6 [11.7, 16.1] | 7.5 [6.4, 8.6] | 0.689 [0.596, 0.808] | 0.506 [0.457, 0.551] | <0.001 | 0.510 | 0.264 | 0.261 | 0.304 |
| 0.5 | 1.20 | 42.0 [39.4, 45.3] | 14.6 [11.6, 17.8] | 2.112 [1.979, 2.281] | 1.099 [0.961, 1.250] | <0.001 | 1.860 | 0.680 | 0.676 | 0.695 |

## Class mix (PPO)

The flat penalty is matched to the average weighted penalty at each mix.

| Critical share (%) | Flat penalty | Flat v_crit (%) | Weighted v_crit (%) | Flat cost | Weighted cost | p cost | DP flat cost | DP weighted cost |
|---|---|---|---|---|---|---|---|---|
| 30 | 4.4 | 25.9 [22.4, 29.8] | 14.6 [10.7, 19.9] | 0.785 [0.684, 0.901] | 0.537 [0.429, 0.686] | 0.005 | 0.400 | 0.208 |
| 50 | 6.0 | 13.6 [11.7, 16.1] | 7.5 [6.4, 8.6] | 0.689 [0.596, 0.808] | 0.506 [0.457, 0.551] | <0.001 | 0.510 | 0.264 |
| 70 | 7.6 | 12.7 [9.3, 17.0] | 7.4 [5.7, 9.5] | 0.894 [0.661, 1.199] | 0.633 [0.523, 0.763] | 0.105 | 0.587 | 0.342 |

## PPO learning rate (5 seeds)

| Learning rate | Flat v_crit (%) | Weighted v_crit (%) | Flat cost | Weighted cost |
|---|---|---|---|---|
| 0.0001 | 50.9 [49.4, 51.6] | 50.2 [48.5, 51.6] | 2.651 [2.538, 2.718] | 2.608 [2.486, 2.712] |
| 0.0003 | 15.1 [11.4, 18.9] | 7.5 [5.9, 9.0] | 0.762 [0.580, 0.951] | 0.496 [0.424, 0.562] |
| 0.001 | 12.6 [10.4, 16.5] | 4.8 [3.5, 6.1] | 0.641 [0.536, 0.829] | 0.404 [0.368, 0.442] |

## After 100,000 training steps (5 seeds)

Values are for the final policy of each run. dqn-scaled is DQN with all rewards multiplied by 0.1, which leaves the optimal policy unchanged.

| Agent | n | v_crit (%) | v_ana (%) | SLA cost |
|---|---|---|---|---|
| dqn-scaled flat | 5 | 10.9 [10.7, 11.1] | 0.2 [0.1, 0.4] | 0.548 [0.540, 0.555] |
| dqn-scaled weighted | 5 | 3.9 [3.1, 4.9] | 12.0 [9.1, 14.3] | 0.314 [0.293, 0.340] |
| dqn flat | 5 | 11.1 [10.2, 12.1] | 1.3 [0.7, 2.0] | 0.566 [0.527, 0.615] |
| dqn weighted | 5 | 14.9 [12.3, 16.9] | 8.0 [6.5, 9.3] | 0.825 [0.705, 0.928] |
| ppo flat | 5 | 11.4 [11.2, 11.5] | 0.3 [0.1, 0.5] | 0.573 [0.567, 0.578] |
| ppo weighted | 5 | 3.1 [2.7, 3.4] | 13.8 [13.1, 14.6] | 0.292 [0.283, 0.301] |

Weighted against flat SLA cost of the final policies (Mann-Whitney U, 5 against 5 seeds), seeds where the weighted agent beats each rule on SLA cost, and distance of the weighted agent to the optimum:

| Agent | Weighted cost | Flat cost | p | Beats reservation | Beats SLA rule | Above optimum |
|---|---|---|---|---|---|---|
| ppo | 0.292 | 0.573 | 0.008 | 4/5 | 5/5 | 12% |
| dqn | 0.825 | 0.566 | 0.016 | 0/5 | 0/5 | 216% |
| dqn-scaled | 0.314 | 0.548 | 0.008 | 2/5 | 5/5 | 20% |

## Check of the exact evaluation

Every learned agent is also simulated on the same 100 held-out episodes. Mean and largest difference between simulation and exact evaluation over all main runs:

| Metric | Mean difference | Largest difference | Runs |
|---|---|---|---|
| critical misses | 0.0210 | 0.0594 | 60 |
| analytics misses | 0.0116 | 0.0372 | 60 |
| SLA cost | 0.1164 | 0.3301 | 60 |

The differences come from the finite test set, not from the exact model. All agents share the same 100 episodes, so they share the same offset. The same offset appears for fixed policies, and it disappears with 3,000 episodes. Critical miss rate (%) of three fixed policies:

| Policy | Exact | 20 episodes (poster protocol) | 100 episodes | 3,000 episodes |
|---|---|---|---|---|
| SLA rule | 11.65 | 10.67 | 12.59 | 11.72 |
| Reservation rule | 1.26 | 1.01 | 1.39 | 1.29 |
| Optimum (DP) | 2.06 | 1.71 | 2.25 | 2.09 |

