import json
import os

from qos_offloading import config as C
from qos_offloading.envs import EdgeCloudEnvScarce
from qos_offloading.baselines import RoundRobinBaseline, ShortestQueueBaseline, SLAThresholdBaseline
from qos_offloading.evaluation import evaluate_policy, summarize, print_results, print_summary
from qos_offloading.training import run_multi_seed_dqn

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "poster_v1")


def save(name, result):
    with open(os.path.join(RESULTS_DIR, name), "w") as f:
        json.dump(result, f, indent=1)


def main():
    sla = {C.CRITICAL: C.CRITICAL_SLA_MS, C.ANALYTICS: C.ANALYTICS_SLA_SCARCE_MS}
    heuristics = {
        "rr": RoundRobinBaseline(),
        "jsq": ShortestQueueBaseline(),
        "threshold": SLAThresholdBaseline(sla),
    }
    for name, policy in heuristics.items():
        r = evaluate_policy(policy, env_cls=EdgeCloudEnvScarce)
        print_results(f"{name} (scarce env)", r)
        save(f"scarce_{name}_0.json", r)

    print("\nTraining DQN (weighted reward) on EdgeCloudEnvScarce...")
    dqn = run_multi_seed_dqn(EdgeCloudEnvScarce, EdgeCloudEnvScarce, C.SEEDS, label="dqn")
    for s, r in zip(C.SEEDS, dqn):
        save(f"scarce_dqn_{s}.json", r)
    print_summary(f"DQN (weighted), {len(C.SEEDS)} seeds", summarize(dqn))

    print("\nTraining tuned DQN (weighted reward) on EdgeCloudEnvScarce...")
    dqn_tuned = run_multi_seed_dqn(EdgeCloudEnvScarce, EdgeCloudEnvScarce, C.SEEDS, label="dqn_tuned",
                                   **C.DQN_TUNED_KWARGS)
    for s, r in zip(C.SEEDS, dqn_tuned):
        save(f"scarce_dqn_tuned_{s}.json", r)
    print_summary(f"DQN tuned (weighted), {len(C.SEEDS)} seeds", summarize(dqn_tuned))


if __name__ == "__main__":
    main()
