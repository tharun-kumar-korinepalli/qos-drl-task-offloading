# QoS-Aware Deep Reinforcement Learning for SLA-Differentiated Task Offloading in the IoT Edge-Cloud Continuum

This repository contains a Gymnasium simulator of task offloading between two edge nodes and a cloud, deep reinforcement learning agents (PPO, DQN and A2C from Stable-Baselines3) trained with a class-weighted SLA reward and with a class-blind reward of the same mean penalty, rule-based baselines, and an exact dynamic programming solver for the same simulator. The solver gives the optimal policy, the optimal trade-off between the two task classes, and an exact evaluation of any policy. The extended study of the journal revision adds a general simulator and exact model with N heterogeneous edge nodes, K task classes, random background traffic and a cloud that meets a limit only by chance, fair hyperparameter tuning of every algorithm, published reward designs rebuilt in the same simulator, and changes of the system. Every number in `RESULTS.md`, `RESULTS_v3.md` and every figure is produced by the code here.

## Contents

```
qos_offloading/
  config.py           constants: latencies, SLA limits, penalties, seeds, budgets
  params.py           EnvParams, one frozen dataclass that describes a setting and a reward
  envs/               simulators; parametric.py holds OffloadingEnv, used by all journal experiments
  dp.py               exact model of the two-node simulator, finite-horizon DP, exact policy evaluation, Pareto frontier
  general.py          general system (N nodes, K classes, random background, cloud hit chance), sparse exact model, rules
  gexp.py             training jobs on the general system with exact evaluation, caching and saved policy tables
  studies_v3.py       every experiment grid of the extended study
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
  reward_analysis.py  exact analysis of the reward (bonus shifts the penalties, operator weight sweep)
  scenario_reference.py  exact references of all four systems (rules, tuned rules, optimum, DP policy of each reward)
  dp_scaling.py       size and time of the exact DP for 2 to 5 edge nodes
  run_v3.py           runs the extended study in stages (tuning, main, larger systems, published designs, changes)
  analyze_v3.py       builds results/summary/summary_v3.json
  report_v3.py        writes RESULTS_v3.md
  legacy/             scripts of the 3-page poster version (results in results/poster_v1)
figures/
  make_journal_figures.py   fig_policies, fig_frontier, fig_sweeps, fig_curves
  make_architecture.py      system diagram
  make_v3_figures.py        figures of the extended study
tests/                unit tests, including checks that the exact model matches the simulator
results/
  runs/               one JSON file per trained agent, with its full specification and library versions
  summary/            summary.json
  tables/             CSV and LaTeX tables
  poster_v1/          per-seed output of the poster version
run_all.py            runs everything in order
```

## Paper

`paper/` holds the journal paper: `main.tex` (one file, compiles with pdflatex on Overleaf), `IEEEtran.cls`, every figure, the section sources in `paper/parts/`, `paper/gen/fill.py` and `fill_results.py`, which write every table from `results/summary/`, and the response to reviewer objections in `paper/response/`.

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

## Extended study (journal revision)

```
python experiments/reward_analysis.py
python experiments/scenario_reference.py
python experiments/dp_scaling.py
for i in 0 1; do python experiments/run_v3.py tune,main,scen,stune,scen2,pub,ft $i/2 & done; wait
python experiments/analyze_v3.py
python experiments/report_v3.py
python figures/make_v3_figures.py
```

`run_v3.py` takes the stages and a shard `i/n`, so several processes can share the work. Finished runs are cached in `results/gruns/`, the policy table of every exactly scored agent in `results/gtables/`, and the nominal PPO networks used for fine-tuning in `results/gmodels/`. The committed results let the last three commands rebuild everything without training. `docs/self_review.md` lists the reviewer objections that the extended study answers.

| Stage | Content | Runs |
|---|---|---|
| tune | PPO, DQN, A2C, 6 configurations each, weighted and flat reward, 2 seeds, 50,000 steps, chosen by exact expected return | 72 |
| main | nominal system, PPO (weighted, corrected, flat), DQN and A2C (weighted, flat), 10 seeds, 100,000 steps, exact curves every 5,000 steps | 70 |
| scen | three larger systems, PPO with the nominal setting, three rewards, 10 seeds | 90 |
| stune | three larger systems, three PPO configurations, weighted and flat, 2 seeds, full budget | 36 |
| scen2 | three larger systems, PPO with the setting chosen per system, three rewards, 10 seeds | 90 |
| pub | tiered adaptive penalties with risk features and action pruning after Yamsani and Chenna Reddy (2026), and an ablation, 10 seeds | 30 |
| ft | five changes of the nominal system, nominal PPO fine-tuned for 25,000 steps, weighted and flat, 10 seeds | 100 |

Runs that share a specification are trained once.

## Studies of the first journal draft

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

## Simulator (nominal system)

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
