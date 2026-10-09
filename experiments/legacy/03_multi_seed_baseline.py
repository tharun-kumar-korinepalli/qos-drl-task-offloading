from qos_offloading import config as C
from qos_offloading.envs import EdgeCloudEnv, EdgeCloudEnvClassBlind
from qos_offloading.evaluation import summarize, print_summary
from qos_offloading.training import run_multi_seed


def main():
    weighted = run_multi_seed(EdgeCloudEnv, EdgeCloudEnv, C.SEEDS, label="weighted")
    blind = run_multi_seed(EdgeCloudEnvClassBlind, EdgeCloudEnv, C.SEEDS, label="class-blind")

    print()
    print_summary(f"QoS-differentiated, {len(C.SEEDS)} seeds", summarize(weighted))
    print()
    print_summary(f"Class-blind, {len(C.SEEDS)} seeds", summarize(blind))


if __name__ == "__main__":
    main()
