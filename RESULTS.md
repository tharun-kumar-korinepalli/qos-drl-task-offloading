# Results

Learned agents: mean over 10 seeds with 95% bootstrap interval. Rules and DP policies: exact. All values come from the exact model unless marked as simulated. SLA cost J = sum_c p_c e_c v_c.

## Exact references

| System | States | Optimum | SLA rule | Reservation | Node sets | DP weighted | DP corrected | DP flat |
|---|---|---|---|---|---|---|---|---|
| Nominal | 32,400 | 0.2614 | 0.5828 (123.0%) | 0.3042 (16.4%) | 0.3042 (16.4%) | 0.2641 (1.0%) | 0.2614 (0.0%) | 0.5103 (95.2%) |
| Hetero3 | 291,600 | 0.0899 | 0.5269 (486.2%) | 0.1415 (57.4%) | 0.1175 (30.7%) | 0.0909 (1.1%) | 0.0900 (0.1%) | 0.1479 (64.6%) |
| Hetero3-3C | 437,400 | 0.0839 | 0.6973 (731.2%) | 0.1584 (88.8%) | 0.1095 (30.6%) | 0.0926 (10.4%) | 0.0839 (0.0%) | 0.1814 (116.2%) |
| Large4-3C | 3,936,600 | 0.0777 | 0.5107 (557.4%) | 0.1024 (31.8%) | 0.0937 (20.6%) | 0.0815 (4.9%) | 0.0777 (0.1%) | 0.1695 (118.2%) |

## Operator weight ratio (nominal system)

| Ratio | Optimum | Corrected penalties | Corrected | Weighted | Flat |
|---|---|---|---|---|---|
| 1 | 0.1021 | 6.00, 6.00 | 0.0% | 0.0% | 0.0% |
| 2 | 0.1746 | 8.33, 3.67 | 0.0% | 0.2% | 17.0% |
| 3 | 0.2128 | 9.50, 2.50 | 0.0% | 0.8% | 43.9% |
| 5 | 0.2614 | 10.67, 1.33 | 0.0% | 1.0% | 95.2% |
| 10 | 0.3435 | 11.73, 0.27 | 0.0% | 1.5% | 197.1% |
| 20 | 0.4743 | 12.33, -0.33 | 0.0% | 8.3% | 330.3% |
| 50 | 0.7400 | 12.73, -0.73 | 1.0% | 26.9% | 589.5% |

## Size and time of the exact DP

| Edge nodes | States | Transitions | Solve (s) | Evaluate (s) | Peak memory (MB) |
|---|---|---|---|---|---|
| 2 | 32,400 | 1,376 | 0.01 | 0.00 | 57 |
| 3 | 291,600 | 32,050 | 0.02 | 0.01 | 59 |
| 4 | 2,624,400 | 775,900 | 0.29 | 0.14 | 92 |
| 5 | 23,619,600 | 11,302,060 | 3.39 | 6.68 | 485 |

## Tuning

- ppo|weighted: configuration 4 {'batch_size': 128, 'ent_coef': 0.0, 'gamma': 0.99, 'learning_rate': 0.001, 'n_steps': 1024, 'net_arch': [128, 128]}, expected return 120.2
- ppo|flat: configuration 4 {'batch_size': 128, 'ent_coef': 0.0, 'gamma': 0.99, 'learning_rate': 0.001, 'n_steps': 1024, 'net_arch': [128, 128]}, expected return 117.5
- dqn|weighted: configuration 2 {'buffer_size': 100000, 'exploration_fraction': 0.1, 'gamma': 0.99, 'learning_rate': 0.0001, 'net_arch': [128, 128], 'reward_scale': 0.1, 'target_update_interval': 2000}, expected return 117.1
- dqn|flat: configuration 2 {'buffer_size': 100000, 'exploration_fraction': 0.1, 'gamma': 0.99, 'learning_rate': 0.0001, 'net_arch': [128, 128], 'reward_scale': 0.1, 'target_update_interval': 2000}, expected return 122.2
- a2c|weighted: configuration 5 {'ent_coef': 0.0, 'gamma': 0.999, 'learning_rate': 0.001, 'n_steps': 5, 'net_arch': [128, 128], 'normalize_advantage': False}, expected return 94.6
- a2c|flat: configuration 0 {}, expected return 43.2
- hetero3|weighted: configuration 4, expected return 162.8
- hetero3|flat: configuration 4, expected return 143.8
- hetero3_3c|weighted: configuration 4, expected return 158.8
- hetero3_3c|flat: configuration 4, expected return 146.7
- large4_3c|weighted: configuration 4, expected return 162.6
- large4_3c|flat: configuration 0, expected return 144.5

## Nominal system, 100,000 steps

| Agent | v_crit (%) | v_ana (%) | SLA cost | Above optimum | Beats reservation | Beats SLA rule |
|---|---|---|---|---|---|---|
| ppo|weighted | 2.9 | 14.4 | 0.291 [0.278, 0.307] | 11.3% | 8/10 (p=0.109) | 10/10 |
| ppo|corrected | 2.7 | 14.8 | 0.284 [0.277, 0.296] | 8.6% | 9/10 (p=0.021) | 10/10 |
| ppo|flat | 11.3 | 0.2 | 0.566 [0.556, 0.577] | 116.7% | 0/10 (p=0.002) | 7/10 |
| dqn|weighted | 4.5 | 10.6 | 0.330 [0.309, 0.352] | 26.3% | 2/10 (p=0.109) | 10/10 |
| dqn|flat | 11.0 | 0.1 | 0.550 [0.540, 0.563] | 110.4% | 0/10 (p=0.002) | 9/10 |
| a2c|weighted | 18.1 | 3.9 | 0.946 [0.678, 1.247] | 261.9% | 0/10 (p=0.002) | 4/10 |
| a2c|flat | 13.5 | 0.2 | 0.678 [0.608, 0.763] | 159.6% | 0/10 (p=0.002) | 3/10 |

| Algo | Rewards | Metric | Diff | A12 | p (Holm) | Lower |
|---|---|---|---|---|---|---|
| ppo | weighted vs flat | cost | -0.2754 | 0.00 | 0.0002 | 10/10 |
| ppo | weighted vs flat | miss0 | -0.0834 | 0.00 | 0.0002 | 10/10 |
| ppo | weighted vs flat | miss1 | +0.1418 | 1.00 | 0.0002 | 0/10 |
| ppo | corrected vs flat | cost | -0.2825 | 0.00 | 0.0002 | 10/10 |
| ppo | corrected vs flat | miss0 | -0.0856 | 0.00 | 0.0002 | 10/10 |
| ppo | corrected vs flat | miss1 | +0.1455 | 1.00 | 0.0002 | 0/10 |
| ppo | corrected vs weighted | cost | -0.0070 | 0.40 | 1.0000 | 6/10 |
| ppo | corrected vs weighted | miss0 | -0.0022 | 0.40 | 1.0000 | 5/10 |
| ppo | corrected vs weighted | miss1 | +0.0038 | 0.69 | 0.8275 | 4/10 |
| dqn | weighted vs flat | cost | -0.2197 | 0.00 | 0.0002 | 10/10 |
| dqn | weighted vs flat | miss0 | -0.0648 | 0.00 | 0.0002 | 10/10 |
| dqn | weighted vs flat | miss1 | +0.1045 | 1.00 | 0.0002 | 0/10 |
| a2c | weighted vs flat | cost | +0.2676 | 0.59 | 1.0000 | 5/10 |
| a2c | weighted vs flat | miss0 | +0.0461 | 0.58 | 1.0000 | 5/10 |
| a2c | weighted vs flat | miss1 | +0.0371 | 1.00 | 0.0002 | 0/10 |

## Larger systems, PPO tuned per system

| System, reward | SLA cost | Above optimum | Beats reservation | Beats node sets |
|---|---|---|---|---|
| hetero3|weighted | 0.1302 [0.1201, 0.1442] | 44.8% | 8/10 | 3/10 |
| hetero3|corrected | 0.1271 [0.1171, 0.1378] | 41.5% | 7/10 | 3/10 |
| hetero3|flat | 0.2126 [0.1707, 0.2622] | 136.5% | 1/10 | 1/10 |
| hetero3_3c|weighted | 0.1410 [0.1344, 0.1483] | 68.1% | 9/10 | 0/10 |
| hetero3_3c|corrected | 0.1303 [0.1241, 0.1366] | 55.3% | 10/10 | 0/10 |
| hetero3_3c|flat | 0.2874 [0.2497, 0.3280] | 242.6% | 0/10 | 0/10 |
| large4_3c|weighted | 0.1366 [0.1255, 0.1494] | 75.8% | 0/10 | 0/10 |
| large4_3c|corrected | 0.1306 [0.1227, 0.1410] | 68.2% | 0/10 | 0/10 |
| large4_3c|flat | 0.3002 [0.2616, 0.3413] | 286.5% | 0/10 | 0/10 |

| System | Rewards | Diff | A12 | p (Holm) | Lower |
|---|---|---|---|---|---|
| hetero3 | weighted vs flat | -0.0824 | 0.12 | 0.0115 | 8/10 |
| hetero3 | corrected vs flat | -0.0855 | 0.08 | 0.0036 | 9/10 |
| hetero3 | corrected vs weighted | -0.0030 | 0.47 | 0.8534 | 5/10 |
| hetero3_3c | weighted vs flat | -0.1464 | 0.00 | 0.0001 | 10/10 |
| hetero3_3c | corrected vs flat | -0.1571 | 0.00 | 0.0001 | 10/10 |
| hetero3_3c | corrected vs weighted | -0.0107 | 0.25 | 0.1890 | 8/10 |
| large4_3c | weighted vs flat | -0.1637 | 0.00 | 0.0001 | 10/10 |
| large4_3c | corrected vs flat | -0.1696 | 0.00 | 0.0001 | 10/10 |
| large4_3c | corrected vs weighted | -0.0059 | 0.37 | 0.7054 | 7/10 |

## Larger systems, PPO with the nominal setting

| System, reward | SLA cost | Above optimum | Beats reservation | Beats node sets |
|---|---|---|---|---|
| hetero3|weighted | 0.1302 [0.1201, 0.1442] | 44.8% | 8/10 | 3/10 |
| hetero3|corrected | 0.1271 [0.1171, 0.1378] | 41.5% | 7/10 | 3/10 |
| hetero3|flat | 0.2126 [0.1707, 0.2622] | 136.5% | 1/10 | 1/10 |
| hetero3_3c|weighted | 0.1410 [0.1344, 0.1483] | 68.1% | 9/10 | 0/10 |
| hetero3_3c|corrected | 0.1303 [0.1241, 0.1366] | 55.3% | 10/10 | 0/10 |
| hetero3_3c|flat | 0.2874 [0.2497, 0.3280] | 242.6% | 0/10 | 0/10 |
| large4_3c|weighted | 0.1366 [0.1255, 0.1494] | 75.8% | 0/10 | 0/10 |
| large4_3c|corrected | 0.1306 [0.1227, 0.1410] | 68.2% | 0/10 | 0/10 |
| large4_3c|flat | 0.2972 [0.2470, 0.3542] | 282.5% | 0/10 | 0/10 |

| System | Rewards | Diff | A12 | p (Holm) | Lower |
|---|---|---|---|---|---|
| hetero3 | weighted vs flat | -0.0824 | 0.12 | 0.0115 | 8/10 |
| hetero3 | corrected vs flat | -0.0855 | 0.08 | 0.0036 | 9/10 |
| hetero3 | corrected vs weighted | -0.0030 | 0.47 | 0.8534 | 5/10 |
| hetero3_3c | weighted vs flat | -0.1464 | 0.00 | 0.0001 | 10/10 |
| hetero3_3c | corrected vs flat | -0.1571 | 0.00 | 0.0001 | 10/10 |
| hetero3_3c | corrected vs weighted | -0.0107 | 0.25 | 0.1890 | 8/10 |
| large4_3c | weighted vs flat | -0.1606 | 0.02 | 0.0003 | 9/10 |
| large4_3c | corrected vs flat | -0.1665 | 0.01 | 0.0002 | 9/10 |
| large4_3c | corrected vs weighted | -0.0059 | 0.37 | 0.7054 | 7/10 |

## Tiered adaptive penalties (nominal system)

| Variant | SLA cost | Above optimum | v_crit (%) | Final penalties | Scored by |
|---|---|---|---|---|---|
| nominal|full|0.001 | 0.360 [0.339, 0.382] | 37.7% | 5.6 | 10.98, 4.89 | 500 episodes |
| nominal|full|0.01 | 0.383 [0.357, 0.417] | 46.6% | 6.3 | 28.82, 15.81 | 500 episodes |
| nominal|adaptive_only|0.01 | 0.414 [0.393, 0.440] | 58.3% | 7.1 | 37.78, 24.07 | exact |

## Changes of the nominal system

| Change | Optimum | SLA rule | Reservation, nominal | Reservation, re-tuned | PPO weighted, nominal | PPO weighted, fine-tuned | PPO flat, nominal | PPO flat, fine-tuned |
|---|---|---|---|---|---|---|---|---|
| slower tasks (+5 ms per queued task) | 0.2615 | 123.1% | 16.3% | 16.3% | 11.2% | 9.2% | 117.6% | 123.2% |
| slower service (d = 0.55) | 0.4421 | 163.3% | 5.6% | 5.6% | 17.5% | 10.5% | 155.6% | 155.2% |
| more background (0.3 per step) | 0.4266 | 168.1% | 6.1% | 6.1% | 18.2% | 15.6% | 161.7% | 160.8% |
| random background (Bernoulli 0.2) | 0.3415 | 106.2% | 6.0% | 6.0% | 6.1% | 7.7% | 99.2% | 99.6% |
| cloud meets analytics limit w.p. 0.5 | 0.1718 | 239.3% | 7.0% | 7.0% | 27.4% | 24.1% | 229.1% | 214.8% |

## Large4-3C, 2M training steps (revision)

| Reward | SLA cost | Above optimum | Above optimum at 200k | p vs 200k (Holm) | Beats node sets |
|---|---|---|---|---|---|
| weighted | 0.1567 [0.1461, 0.1690] | 101.8% | 75.8% | 0.071 | 0/10 |
| corrected | 0.1356 [0.1279, 0.1433] | 74.5% | 68.2% | 0.315 | 0/10 |

## PPO-Lagrangian, targets = miss rates of the optimum (revision)

| System | Step size | SLA cost | Above optimum | Seeds meeting all targets | p vs weighted PPO (Holm) | Final multipliers |
|---|---|---|---|---|---|---|
| nominal | 1.0 | 0.2863 [0.2804, 0.2929] | 9.5% | 1/10 | 0.739 | 13.69, 2.00 |
| large4_3c | 0.1 | 0.1304 [0.1192, 0.1428] | 67.8% | 0/10 | 0.705 | 11.00, 4.72, 1.00 |

## Runtimes (Intel(R) Core(TM) Ultra 9 185H, one core)

| System | DP solve (s) | Exact evaluation (s) | PPO steps | PPO training (min) |
|---|---|---|---|---|
| Nominal | 0.00 | 0.00 | 100,352 | 0.6 |
| Hetero3 | 0.02 | 0.01 | 100,352 | 0.6 |
| Hetero3-3C | 0.02 | 0.01 | 100,352 | 0.6 |
| Large4-3C | 0.26 | 0.13 | 200,704 | 1.2 |
