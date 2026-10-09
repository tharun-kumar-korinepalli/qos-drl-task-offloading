import json
import os

from qos_offloading import config as C
from qos_offloading.envs import EdgeCloudEnvScarce
from qos_offloading.envs.scarce import make_scarce_env_with_penalty
from qos_offloading.evaluation import summarize, print_summary
from qos_offloading.training import run_multi_seed

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "poster_v1")
PENALTY_PAIRS = [(6, 6), (8, 4), (10, 2), (11, 1)]


def main():
    for w_crit, w_ana in PENALTY_PAIRS:
        label = f"sweep_{w_crit}_{w_ana}"
        train_env = make_scarce_env_with_penalty(w_crit, w_ana)
        results = run_multi_seed(train_env, EdgeCloudEnvScarce, C.SEEDS, label=label)
        for s, r in zip(C.SEEDS, results):
            with open(os.path.join(RESULTS_DIR, f"scarce_{label}_{s}.json"), "w") as f:
                json.dump(r, f, indent=1)
        print_summary(f"{label}, {len(C.SEEDS)} seeds", summarize(results))


if __name__ == "__main__":
    main()
