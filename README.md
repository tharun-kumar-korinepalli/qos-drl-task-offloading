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
scripts/        run, analyze and plot
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
