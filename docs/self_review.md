# Self review of the journal draft

This is a hard review of the first journal draft, written the way an experienced IEEE reviewer would write it. For each point, the right column says what the revision does about it. Results are filled in from `results/` once the runs finish.

## Major points

| # | Objection | Answer in the revision |
|---|---|---|
| M1 | The system is tiny. If DP solves it exactly, why use deep RL at all? | Three larger systems (3 and 4 heterogeneous edge nodes, 2 and 3 classes, random background, a cloud that sometimes meets the analytics limit) that DP still solves exactly. DP time and memory grow as 9^N, which shows where exact solution stops. A model-shift study shows what happens to rules built for one system when the system changes. |
| M2 | A tuned reservation rule beats every agent. | Main study moves to 100k steps after a fair tuning budget. A second, richer rule family (per-class node sets and thresholds) is tuned on the exact model, so the rules get the strongest version available. Results are reported whether the agents win or lose. |
| M3 | A2C fails, DQN diverges. Is this a fair test of the algorithms? | Same tuning budget for every algorithm and reward (6 configurations, 2 seeds, 50k steps, chosen by each agent's own expected return, never by the SLA cost). Reward scale is part of the DQN search. |
| M4 | No comparison with a published method. | Flat penalty with the class in the state is the design of Nieto et al. (2024). The tiered reward with adaptive penalties, risk features and action pruning of Yamsani and Chenna Reddy (2026) is rebuilt in the same simulator, with an ablation that keeps only the adaptive penalties. |
| M5 | The SLA cost uses the same weights as the reward, so the comparison is circular. | Exact sweep of the operator's weight ratio from 1 to 50. For each ratio, the optimum, the matched weighted reward and the flat reward are computed exactly. Pareto frontier and per-class miss rates stay in the paper. |
| M6 | Why is the weighted reward 1% from the optimum and not exactly at it? | The +1 reward for a met limit adds 1 to every penalty, so 10:2 acts as 11:3. Folding the bonus into the penalties (corrected reward) gives the optimum exactly. This is stated as a proposition, checked by DP, and the corrected reward is trained as a third reward. |
| M7 | A cloud that always misses is not a real offloading choice. | In the larger systems and in one shift the cloud meets the analytics limit with a given probability. |
| M8 | Deterministic background traffic and a linear latency model are unrealistic. | Random (Bernoulli) background in all larger systems and as a shift. Real traces are not available to the authors, which is stated as a limitation. |
| M9 | 20k steps is an arbitrary budget. | 100k steps for the main study, with exact learning curves every 5k steps over 10 seeds. The 20k results stay as a low-budget view. |
| M10 | Agents and rules are only tested on the system they were built for. | Seven shifts of the nominal system. Rules keep their nominal model and thresholds, agents keep their nominal policy, and both are scored exactly. Then rules are re-tuned with the true model, agents are fine-tuned for 25k steps from interaction alone, and the new optimum is computed. |

## Minor points

| # | Objection | Answer |
|---|---|---|
| m1 | Agents use a discount of 0.99, the objective is undiscounted. | Exact effect of the discount on the reward-optimal policy. |
| m2 | Curves used 5 seeds. | 10 seeds. |
| m3 | Does the agent need the time feature? | Exact cost of the best time-invariant use of the DP policy. |
| m4 | Decision time of the agent. | Measured per decision, against a table lookup. |
| m5 | Statistics for 5-seed comparisons are weak. | All main comparisons now use 10 seeds. Sign tests and Holm correction kept. |
