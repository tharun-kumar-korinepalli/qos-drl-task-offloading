import json
import os

from qos_offloading import config as C
from qos_offloading.envs import EdgeCloudEnvScarce, EdgeCloudEnvScarceClassBlind
from qos_offloading.baselines import RoundRobinBaseline
from qos_offloading.evaluation import evaluate_policy, summarize, print_results, print_summary
from qos_offloading.training import run_multi_seed

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "poster_v1")


def save(name, result):
    with open(os.path.join(RESULTS_DIR, name), "w") as f:
        json.dump(result, f, indent=1)


def main():
    rr_results = evaluate_policy(RoundRobinBaseline(), env_cls=EdgeCloudEnvScarce)
    print_results("Round-robin (scarce env)", rr_results)
    save("scarce_rr_0.json", rr_results)

    print("\nTraining WEIGHTED (QoS-differentiated) agent on EdgeCloudEnvScarce...")
    weighted = run_multi_seed(EdgeCloudEnvScarce, EdgeCloudEnvScarce, C.SEEDS, label="weighted")
    for s, r in zip(C.SEEDS, weighted):
        save(f"scarce_weighted_{s}.json", r)

    print("\nTraining FLAT (class-blind) agent on EdgeCloudEnvScarce...")
    flat = run_multi_seed(EdgeCloudEnvScarceClassBlind, EdgeCloudEnvScarce, C.SEEDS, label="flat")
    for s, r in zip(C.SEEDS, flat):
        save(f"scarce_flat_{s}.json", r)

    print()
    print_results("Round-robin (scarce env)", rr_results)
    print()
    print_summary(f"WEIGHTED (QoS-differentiated), {len(C.SEEDS)} seeds", summarize(weighted))
    print()
    print_summary(f"FLAT (class-blind), {len(C.SEEDS)} seeds", summarize(flat))

    print("\nPer-seed critical-task violation rates:")
    print("  weighted:", [round(r["class_0_violation_rate"], 4) for r in weighted])
    print("  flat:    ", [round(r["class_0_violation_rate"], 4) for r in flat])
    print("Per-seed analytics-task violation rates:")
    print("  weighted:", [round(r["class_1_violation_rate"], 4) for r in weighted])
    print("  flat:    ", [round(r["class_1_violation_rate"], 4) for r in flat])


if __name__ == "__main__":
    main()
