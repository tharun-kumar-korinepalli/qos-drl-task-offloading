# QoS-Aware Deep Reinforcement Learning for SLA-Differentiated Task Offloading in the IoT Edge-Cloud Continuum

This repository contains a Gymnasium simulator of task offloading between two edge nodes and a cloud, deep reinforcement learning agents (PPO, DQN and A2C from Stable-Baselines3) trained with a class-weighted SLA reward and with a class-blind reward of the same mean penalty, rule-based baselines, and an exact dynamic programming solver for the same simulator. The solver gives the optimal policy, the optimal trade-off between the two task classes, and an exact evaluation of any policy. Every number in `RESULTS.md` and every figure is produced by the code here. `RESULTS.md` lists all results.

## Contents

```
qos_offloading/
  config.py           constants: latencies, SLA limits, penalties, seeds, budgets
  params.py           EnvParams, one frozen dataclass that describes a setting and a reward
  envs/               simulators; parametric.py holds OffloadingEnv, used by all journal experiments
  dp.py               exact model of the simulator, finite-horizon DP, exact policy evaluation, Pareto frontier
  reference.py        rule baselines and DP reference policies for any setting
  baselines.py        round-robin, join-shortest-queue, SLA rule, reservation rule, table policy
  agents.py           PPO, DQN, A2C training with exact and Monte Carlo evaluation, learning curves
  runner.py           job description, content-hashed result cache, parallel execution
  studies.py          every experiment grid of the study
  stats.py            bootstrap intervals, Mann-Whitney U, Vargha-Delaney A12, sign test, Holm correction
  evaluation.py       Monte Carlo evaluation protocol
experiments/
  01_sanity_check.py  Gymnasium and Stable-Baselines3 environment checks
  run.py              trains and evaluates every job of the chosen studies
  analyze.py          builds results/summary/summary.json from all runs
  report.py           writes RESULTS.md and the LaTeX tables in results/tables
  legacy/             scripts of the 3-page poster version (results in results/poster_v1)
figures/
  make_journal_figures.py   fig_policies, fig_frontier, fig_sweeps, fig_curves
  make_architecture.py      system diagram
tests/                unit tests, including checks that the exact model matches the simulator
results/
  runs/               one JSON file per trained agent, with its full specification and library versions
  summary/            summary.json
  tables/             CSV and LaTeX tables
  poster_v1/          per-seed output of the poster version
run_all.py            runs everything in order
```

## Setup

Python 3.10 or newer. The experiments run on CPU.

```
git clone https://github.com/tharun-kumar-korinepalli/qos-drl-task-offloading.git
cd qos-drl-task-offloading
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest
```

On Windows replace `source .venv/bin/activate` with `.venv\Scripts\activate`.

## Reproducing all results

```
python run_all.py 4
```

The argument is the number of worker processes. The full study is 260 training runs. On two CPU cores it takes about 85 minutes, and each worker uses one thread. Results are cached in `results/runs/`, keyed by a hash of the job specification, so an interrupted run continues where it stopped and finished jobs are never trained twice. The committed `results/runs/` already holds every run, so `python experiments/analyze.py`, `python experiments/report.py` and `python figures/make_journal_figures.py` rebuild the summary, tables and figures in under a minute without any training.

Single studies can be run on their own:

```
python experiments/run.py main --workers 2
python experiments/run.py penalty load mix lr curves dqn_scaled --workers 2
```

## Studies

| Study | Content | Runs |
|---|---|---|
| main | PPO, DQN, A2C, each with weighted and flat reward, 10 seeds, 20,000 steps | 60 |
| penalty | PPO, penalty ratio 6:6, 8:4, 9:3, 10:2, 11:1 at a mean of 6, 10 seeds | 50 |
| load | PPO, weighted and flat, drain probability 0.9 to 0.5 (offered edge load 0.67 to 1.2), 10 seeds | 100 |
| mix | PPO, weighted and flat, share of critical tasks 30, 50, 70%, flat penalty matched to the mix, 10 seeds | 60 |
| lr | PPO, learning rate 1e-4, 3e-4, 1e-3, 5 seeds | 30 |
| curves | PPO and DQN, weighted and flat, 100,000 steps, exact evaluation every 5,000 steps, 5 seeds | 20 |
| dqn_scaled | DQN as in curves with every reward multiplied by 0.1, which leaves the optimal policy unchanged | 10 |

Runs shared between studies are trained once.

## Evaluation

The state of the simulator is the two queue lengths (0 to 8), the class of the arriving task and the share of the episode left. Because the background traffic is deterministic in time and the episode length is fixed, the share of the episode left determines the full state, so the observation is Markov. `dp.py` enumerates all 200 x 81 x 2 states, builds the exact transition matrices, and propagates the state distribution forward under any deterministic policy. This gives the expected miss rate of each class, mean latency and edge use without sampling noise. Trained agents are queried once on every state. Every agent is also simulated on 100 held-out episodes as a check, and the difference between the two evaluations is reported in `RESULTS.md`. The same model gives the optimal policy for the SLA cost, the optimal policy for each training reward, and the Pareto frontier of critical against analytics misses.

## Simulator

| Quantity | Value |
|---|---|
| Edge latency | 30 ms + 15 ms per queued task, queue capped at 8 |
| Cloud latency | 110 ms, misses both limits |
| Latency limit | critical 50 ms, analytics 90 ms |
| Arrivals | 1 agent task per step plus 0.2 background tasks per step to the shorter queue |
| Service | each edge queue finishes one task per step with probability 0.6 |
| Reward | +1 if the limit is met, minus the class penalty if missed, minus 0.1 for every cloud placement |
| Weighted penalty | 10 critical, 2 analytics |
| Flat penalty | the mean of the weighted penalties at the class mix, 6 at a 50/50 mix |
| Episode | 200 steps |

## Reproducibility

Training seeds, Monte Carlo evaluation seeds and the bootstrap seed are fixed. With the pinned versions in `requirements.txt` the poster results in `results/poster_v1` are reproduced exactly by `experiments/legacy/05_scarce.py`. Each file in `results/runs/` records the Python, NumPy, Gymnasium, Stable-Baselines3 and PyTorch versions it was produced with. Other library versions change the training path of the agents but not the exact solver.
