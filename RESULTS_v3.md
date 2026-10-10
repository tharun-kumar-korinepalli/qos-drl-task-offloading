# Results of the extended study (v3)

Produced by `experiments/run_v3.py`, `experiments/analyze_v3.py` and this script. Learned agents: mean over 10 seeds with 95% bootstrap interval. Rules and DP policies: exact. Every value comes from the exact model unless marked as simulated. SLA cost J = sum_c p_c e_c v_c.

## Exact references

| System | States | Optimum | SLA rule | Reservation | Reservation, node sets | DP weighted | DP corrected | DP flat |
|---|---|---|---|---|---|---|---|---|
| Nominal | 32,400 | 0.2614 | 0.5828 (123.0%) | 0.3042 (16.4%) | 0.3042 (16.4%) | 0.2641 (1.0%) | 0.2614 (0.0%) | 0.5103 (95.2%) |
| Hetero3 | 291,600 | 0.0899 | 0.5269 (486.2%) | 0.1415 (57.4%) | 0.1175 (30.7%) | 0.0909 (1.1%) | 0.0900 (0.1%) | 0.1479 (64.6%) |
| Hetero3-3C | 437,400 | 0.0839 | 0.6973 (731.2%) | 0.1584 (88.8%) | 0.1095 (30.6%) | 0.0926 (10.4%) | 0.0839 (0.0%) | 0.1814 (116.2%) |
| Large4-3C | 3,936,600 | 0.0777 | 0.5107 (557.4%) | 0.1024 (31.8%) | 0.0937 (20.6%) | 0.0815 (4.9%) | 0.0777 (0.1%) | 0.1695 (118.2%) |

## Operator weight ratio (nominal system, exact)

| Ratio | Optimum | Corrected penalties | Corrected | Weighted | Flat |
|---|---|---|---|---|---|
| 1 | 0.1021 | 6.00, 6.00 | 0.0% | 0.0% | 0.0% |
| 2 | 0.1746 | 8.33, 3.67 | 0.0% | 0.2% | 17.0% |
| 3 | 0.2128 | 9.50, 2.50 | 0.0% | 0.8% | 43.9% |
| 5 | 0.2614 | 10.67, 1.33 | 0.0% | 1.0% | 95.2% |
| 10 | 0.3435 | 11.73, 0.27 | 0.0% | 1.5% | 197.1% |
| 20 | 0.4743 | 12.33, -0.33 | 0.0% | 8.3% | 330.3% |
| 50 | 0.7400 | 12.73, -0.73 | 1.0% | 26.9% | 589.5% |

## DP size and time

| Edge nodes | States | Transitions | Solve (s) | Evaluate (s) | Peak memory (MB) |
|---|---|---|---|---|---|
| 2 | 32,400 | 1,376 | 0.02 | 0.02 | 51 |
| 3 | 291,600 | 32,050 | 0.19 | 0.08 | 53 |
| 4 | 2,624,400 | 775,900 | 1.21 | 0.59 | 83 |
| 5 | 23,619,600 | 11,302,060 | 15.71 | 25.89 | 477 |

## Tuning (nominal system)

- ppo|weighted: configuration 4 {'batch_size': 128, 'ent_coef': 0.0, 'gamma': 0.99, 'learning_rate': 0.001, 'n_steps': 1024, 'net_arch': [128, 128]}, mean expected return 118.0
- ppo|flat: configuration 4 {'batch_size': 128, 'ent_coef': 0.0, 'gamma': 0.99, 'learning_rate': 0.001, 'n_steps': 1024, 'net_arch': [128, 128]}, mean expected return 116.9
- dqn|weighted: configuration 2 {'buffer_size': 100000, 'exploration_fraction': 0.1, 'gamma': 0.99, 'learning_rate': 0.0001, 'net_arch': [128, 128], 'reward_scale': 0.1, 'target_update_interval': 2000}, mean expected return 117.0
- dqn|flat: configuration 2 {'buffer_size': 100000, 'exploration_fraction': 0.1, 'gamma': 0.99, 'learning_rate': 0.0001, 'net_arch': [128, 128], 'reward_scale': 0.1, 'target_update_interval': 2000}, mean expected return 122.2
- a2c|weighted: configuration 5 {'ent_coef': 0.0, 'gamma': 0.999, 'learning_rate': 0.001, 'n_steps': 5, 'net_arch': [128, 128], 'normalize_advantage': False}, mean expected return 61.9
- a2c|flat: configuration 5 {'ent_coef': 0.0, 'gamma': 0.999, 'learning_rate': 0.001, 'n_steps': 5, 'net_arch': [128, 128], 'normalize_advantage': False}, mean expected return 58.9

Per-system PPO tuning (configurations 4, 0, 5):

- hetero3|weighted: configuration 0, mean expected return 168.0
- hetero3|flat: configuration 0, mean expected return 150.0
- hetero3_3c|weighted: configuration 4, mean expected return 159.2
- hetero3_3c|flat: configuration 4, mean expected return 146.1
- large4_3c|weighted: configuration 0, mean expected return 162.1
- large4_3c|flat: configuration 0, mean expected return 144.8

## Nominal system, 100,000 steps

| Agent | v_crit (%) | v_ana (%) | SLA cost | Above optimum | Beats reservation | Beats SLA rule |
|---|---|---|---|---|---|---|
| ppo|weighted | 2.7 | 15.4 | 0.287 [0.279, 0.299] | 9.9% | 9/10 (p=0.021) | 10/10 |
| ppo|corrected | 2.6 | 15.7 | 0.286 [0.279, 0.298] | 9.6% | 9/10 (p=0.021) | 10/10 |
| ppo|flat | 11.7 | 0.2 | 0.588 [0.562, 0.625] | 125.1% | 0/10 (p=0.002) | 6/10 |
| dqn|weighted | 4.8 | 10.1 | 0.339 [0.314, 0.361] | 29.5% | 3/10 (p=0.344) | 10/10 |
| dqn|flat | 11.0 | 0.2 | 0.550 [0.537, 0.569] | 110.5% | 0/10 (p=0.002) | 9/10 |
| a2c|weighted | 15.7 | 4.0 | 0.823 [0.614, 1.042] | 214.8% | 0/10 (p=0.002) | 5/10 |
| a2c|flat | 17.4 | 0.4 | 0.872 [0.678, 1.145] | 233.7% | 0/10 (p=0.002) | 2/10 |

| Algo | Rewards | Metric | Diff | A12 | p (Holm) | Lower |
|---|---|---|---|---|---|---|
| ppo | weighted vs flat | cost | -0.3011 | 0.00 | 0.0002 | 10/10 |
| ppo | weighted vs flat | miss0 | -0.0907 | 0.00 | 0.0002 | 10/10 |
| ppo | weighted vs flat | miss1 | +0.1524 | 1.00 | 0.0002 | 0/10 |
| ppo | corrected vs flat | cost | -0.3019 | 0.00 | 0.0002 | 10/10 |
| ppo | corrected vs flat | miss0 | -0.0914 | 0.00 | 0.0002 | 10/10 |
| ppo | corrected vs flat | miss1 | +0.1550 | 1.00 | 0.0002 | 0/10 |
| ppo | corrected vs weighted | cost | -0.0008 | 0.52 | 1.0000 | 6/10 |
| ppo | corrected vs weighted | miss0 | -0.0007 | 0.60 | 1.0000 | 4/10 |
| ppo | corrected vs weighted | miss1 | +0.0027 | 0.47 | 1.0000 | 5/10 |
| dqn | weighted vs flat | cost | -0.2117 | 0.00 | 0.0002 | 10/10 |
| dqn | weighted vs flat | miss0 | -0.0622 | 0.00 | 0.0002 | 10/10 |
| dqn | weighted vs flat | miss1 | +0.0991 | 1.00 | 0.0002 | 0/10 |
| a2c | weighted vs flat | cost | -0.0495 | 0.45 | 1.0000 | 4/10 |
| a2c | weighted vs flat | miss0 | -0.0172 | 0.39 | 1.0000 | 6/10 |
| a2c | weighted vs flat | miss1 | +0.0363 | 0.96 | 0.0008 | 0/10 |

## Larger systems, PPO tuned per system

| System, reward | SLA cost | Above optimum | Beats reservation | Beats node-set rule |
|---|---|---|---|---|
| hetero3|weighted | 0.1288 [0.1163, 0.1445] | 43.3% | 7/10 | 4/10 (p=0.754) |
| hetero3|corrected | 0.1227 [0.1162, 0.1300] | 36.5% | 9/10 | 2/10 (p=0.109) |
| hetero3|flat | 0.1577 [0.1471, 0.1692] | 75.4% | 2/10 | 0/10 (p=0.002) |
| hetero3_3c|weighted | 0.1403 [0.1305, 0.1505] | 67.2% | 8/10 | 0/10 (p=0.002) |
| hetero3_3c|corrected | 0.1400 [0.1248, 0.1611] | 66.9% | 8/10 | 0/10 (p=0.002) |
| hetero3_3c|flat | 0.2981 [0.2534, 0.3460] | 255.4% | 0/10 | 0/10 (p=0.002) |
| large4_3c|weighted | 0.1251 [0.1189, 0.1328] | 61.0% | 0/10 | 0/10 (p=0.002) |
| large4_3c|corrected | 0.1195 [0.1160, 0.1234] | 53.9% | 0/10 | 0/10 (p=0.002) |
| large4_3c|flat | 0.3403 [0.3000, 0.3818] | 338.1% | 0/10 | 0/10 (p=0.002) |

| System | Rewards | Diff | A12 | p (Holm) | Lower |
|---|---|---|---|---|---|
| hetero3 | weighted vs flat | -0.0288 | 0.16 | 0.0357 | 8/10 |
| hetero3 | corrected vs flat | -0.0350 | 0.04 | 0.0006 | 10/10 |
| hetero3 | corrected vs weighted | -0.0062 | 0.50 | 1.0000 | 6/10 |
| hetero3_3c | weighted vs flat | -0.1578 | 0.00 | 0.0001 | 10/10 |
| hetero3_3c | corrected vs flat | -0.1581 | 0.01 | 0.0001 | 10/10 |
| hetero3_3c | corrected vs weighted | -0.0003 | 0.40 | 0.9625 | 6/10 |
| large4_3c | weighted vs flat | -0.2152 | 0.00 | 0.0001 | 10/10 |
| large4_3c | corrected vs flat | -0.2208 | 0.00 | 0.0001 | 10/10 |
| large4_3c | corrected vs weighted | -0.0055 | 0.36 | 0.9450 | 6/10 |

## Larger systems, PPO with the nominal setting

| System, reward | SLA cost | Above optimum | Beats reservation | Beats node-set rule |
|---|---|---|---|---|
| hetero3|weighted | 0.1343 [0.1196, 0.1543] | 49.5% | 8/10 | 3/10 (p=0.344) |
| hetero3|corrected | 0.1165 [0.1136, 0.1195] | 29.6% | 10/10 | 7/10 (p=0.344) |
| hetero3|flat | 0.1996 [0.1712, 0.2319] | 122.1% | 1/10 | 0/10 (p=0.002) |
| hetero3_3c|weighted | 0.1403 [0.1305, 0.1505] | 67.2% | 8/10 | 0/10 (p=0.002) |
| hetero3_3c|corrected | 0.1400 [0.1248, 0.1611] | 66.9% | 8/10 | 0/10 (p=0.002) |
| hetero3_3c|flat | 0.2981 [0.2534, 0.3460] | 255.4% | 0/10 | 0/10 (p=0.002) |
| large4_3c|weighted | 0.1617 [0.1326, 0.1997] | 108.1% | 0/10 | 0/10 (p=0.002) |
| large4_3c|corrected | 0.1334 [0.1197, 0.1507] | 71.8% | 0/10 | 0/10 (p=0.002) |
| large4_3c|flat | 0.3457 [0.2694, 0.4254] | 345.1% | 0/10 | 0/10 (p=0.002) |

| System | Rewards | Diff | A12 | p (Holm) | Lower |
|---|---|---|---|---|---|
| hetero3 | weighted vs flat | -0.0653 | 0.07 | 0.0024 | 9/10 |
| hetero3 | corrected vs flat | -0.0831 | 0.00 | 0.0001 | 10/10 |
| hetero3 | corrected vs weighted | -0.0178 | 0.27 | 0.2676 | 8/10 |
| hetero3_3c | weighted vs flat | -0.1578 | 0.00 | 0.0001 | 10/10 |
| hetero3_3c | corrected vs flat | -0.1581 | 0.01 | 0.0002 | 10/10 |
| hetero3_3c | corrected vs weighted | -0.0003 | 0.40 | 0.4813 | 6/10 |
| large4_3c | weighted vs flat | -0.1841 | 0.08 | 0.0029 | 9/10 |
| large4_3c | corrected vs flat | -0.2123 | 0.02 | 0.0003 | 10/10 |
| large4_3c | corrected vs weighted | -0.0282 | 0.32 | 0.3806 | 6/10 |

## Published reward designs (nominal system)

| Variant | SLA cost | Above optimum | v_crit (%) | Final penalties | Scored by |
|---|---|---|---|---|---|
| nominal|full|0.001 | 0.368 [0.346, 0.393] | 40.9% | 5.9 | 11.10, 4.67 | mc |
| nominal|full|0.01 | 0.418 [0.380, 0.453] | 59.9% | 7.3 | 30.04, 14.14 | mc |
| nominal|adaptive_only|0.01 | 0.414 [0.396, 0.434] | 58.2% | 7.1 | 38.09, 23.05 | exact |

## Changes of the nominal system (exact)

| Change | Optimum | SLA rule | Reservation, nominal | Reservation, re-tuned | PPO weighted, nominal | PPO weighted, fine-tuned | PPO flat, nominal | PPO flat, fine-tuned |
|---|---|---|---|---|---|---|---|---|
| slower tasks (+5 ms per queued task) | 0.2615 | 123.1% | 16.3% | 16.3% | 9.9% | 10.0% | 126.8% | 119.8% |
| slower service (d = 0.55) | 0.4421 | 163.3% | 5.6% | 5.6% | 14.1% | 14.0% | 166.4% | 157.7% |
| more background (0.3 per step) | 0.4266 | 168.1% | 6.1% | 6.1% | 14.9% | 12.0% | 172.3% | 167.0% |
| random background (Bernoulli 0.2) | 0.3415 | 106.2% | 6.0% | 6.0% | 4.3% | 4.9% | 106.6% | 99.5% |
| cloud meets analytics limit w.p. 0.5 | 0.1718 | 239.3% | 7.0% | 7.0% | 22.3% | 19.5% | 242.1% | 233.7% |
