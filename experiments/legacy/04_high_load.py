from qos_offloading import config as C
from qos_offloading.envs import EdgeCloudEnvHighLoad, EdgeCloudEnvHighLoadClassBlind
from qos_offloading.evaluation import summarize, print_summary
from qos_offloading.training import run_multi_seed


def main():
    print("Training QoS-differentiated agent on high-load environment...")
    diff_hl = run_multi_seed(EdgeCloudEnvHighLoad, EdgeCloudEnvHighLoad, C.SEEDS, label="weighted")

    print("Training class-blind agent on high-load environment...")
    blind_hl = run_multi_seed(EdgeCloudEnvHighLoadClassBlind, EdgeCloudEnvHighLoad, C.SEEDS,
                              label="class-blind")

    print()
    print_summary("QoS-differentiated (high load)", summarize(diff_hl))
    print()
    print_summary("Class-blind (high load)", summarize(blind_hl))


if __name__ == "__main__":
    main()
