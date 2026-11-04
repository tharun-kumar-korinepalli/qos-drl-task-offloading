# Class-Weighted Rewards for SLA-Differentiated Task Offloading

IoT tasks of different classes are placed on edge nodes or the cloud. PPO, DQN and A2C agents learn the placement with a class-weighted, a corrected or a class-blind reward. The systems are small enough to solve exactly with dynamic programming, so every agent and every rule is scored against the optimal policy.

## Structure

```
qos_offloading/
  system.py     systems, rewards, Gymnasium environment
  exact.py      exact model: DP solver, exact evaluation of any policy, Pareto frontier
  rules.py      rule baselines and their tuning on the exact model
  training.py   training, exact scoring of trained agents, simulation
  study.py      systems, rewards and experiment grids
  stats.py      bootstrap intervals and statistical tests
scripts/        run, analyze and plot (analyze_revision.py and runtimes.py for the revision)
tests/
results/        trained agents and summaries
figures/      paper figures; Fig. 1 and 2 are TikZ sources (.tex)
RESULTS.md      all result tables
```

## Setup

Python 3.10 or newer, CPU only.

```
git clone https://github.com/tharun-kumar-korinepalli/qos-drl-task-offloading.git
cd qos-drl-task-offloading
pip install -r requirements.txt
pip install -e .
pytest
```

## Reproduce

All trained agents are in `results/`, so the analysis runs without training:

```
python scripts/references.py
python scripts/reward_analysis.py
python scripts/dp_scaling.py
python scripts/analyze.py
python scripts/report.py
python scripts/figures.py
python scripts/analyze_revision.py
```

`scripts/figures.py` draws Fig. 3 to 5 of the paper. The two diagrams, Fig. 1 and Fig. 2, are built with LaTeX:

```
cd figures
pdflatex architecture.tex
pdflatex fig_overview.tex
```

To train everything again, delete `results/runs`, `results/tables` and `results/models`, then run one shard per CPU core:

```
python scripts/run.py all 0/2 & python scripts/run.py all 1/2 & wait
```

Finished runs are cached, so a stopped run continues where it left off. All runs take about 12 hours on two cores.

## Revision experiments

Three additions answer the reviewers. They reuse every stored run, so only the new runs are trained.

| Stage | What it runs | Runs | Approx. cost |
|---|---|---|---|
| `lagrange_tune` | PPO-Lagrangian, step size 0.1, 1 or 10, tuning seeds 100 and 101, nominal and Large4-3C | 12 | 1 core-hour |
| `lagrange` | PPO-Lagrangian with the chosen step size, seeds 0 to 9, nominal (100k steps) and Large4-3C (200k steps) | 20 | 2 core-hours |
| `long` | Weighted and corrected PPO in Large4-3C for 2M steps, seeds 0 to 9, scored every 50k steps | 20 | 15 core-hours |

The PPO-Lagrangian agent treats every class except the least critical one as a constraint, with the miss rates of the exact optimum as targets, so the targets can always be met. After each PPO rollout block, each multiplier takes one dual step, lambda_c = max(0, lambda_c + eta (v_hat_c - target_c)). The step size is chosen by the exact SLA cost on the tuning seeds.

Run them with one shard per core, then measure the runtimes and print the numbers for the paper:

```
python scripts/run.py revision 0/4 & python scripts/run.py revision 1/4 & \
python scripts/run.py revision 2/4 & python scripts/run.py revision 3/4 & wait
python scripts/runtimes.py
python scripts/analyze_revision.py
```

`analyze_revision.py` writes `results/summary/revision.json` and prints every value that the paper reports for these experiments. `runtimes.py` writes `results/summary/runtimes.json` with the CPU model, the DP solve and evaluation time, and the time to train one PPO agent per system. New run records also store `learn_seconds_without_curve`, the training time without the exact scoring of the learning curve.

## Systems

| | Nominal | Hetero3 | Hetero3-3C | Large4-3C |
|---|---|---|---|---|
| Edge nodes, classes | 2, 2 | 3, 2 | 3, 3 | 4, 3 |
| Processing time (ms) | 30, 30 | 20, 30, 40 | 20, 30, 40 | 20, 25, 30, 40 |
| Wait per queued task (ms) | 15, 15 | 10, 15, 20 | 10, 15, 20 | 10, 12, 15, 20 |
| Service probability | 0.6, 0.6 | 0.7, 0.6, 0.5 | 0.7, 0.6, 0.5 | 0.7, 0.6, 0.6, 0.5 |
| Background tasks per step | 0.2, fixed | 0.8, random | 0.8, random | 1.4, random |
| Latency limits (ms) | 50, 90 | 50, 90 | 40, 70, 110 | 40, 70, 110 |
| Class shares | 0.5, 0.5 | 0.5, 0.5 | 0.3, 0.3, 0.4 | 0.3, 0.3, 0.4 |
| Operator weights | 10, 2 | 10, 2 | 10, 4, 1 | 10, 4, 1 |
| Cloud meets limit with probability | 0, 0 | 0, 0.5 | 0, 0.2, 0.6 | 0, 0.2, 0.6 |

Each step one task arrives. The agent sees the queue lengths, the class of the task and the share of the episode left, and places the task on an edge node or the cloud. Edge latency is the processing time plus the wait per queued task times the queue length. The reward is +1 for a met limit, minus the class penalty for a miss, and minus 0.1 for a cloud placement. The SLA cost is the expected operator weight of a missed task.

## Authors

1. Shilpa Golla
2. Tharun Kumar Korine Palli
3. Urmila Bandaru
