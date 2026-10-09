from qos_offloading import config as C
from qos_offloading.envs import EdgeCloudEnv
from qos_offloading.baselines import RoundRobinBaseline
from qos_offloading.evaluation import evaluate_policy, print_results
from qos_offloading.training import train_ppo, make_policy


def main():
    model = train_ppo(EdgeCloudEnv, total_timesteps=C.TOTAL_TIMESTEPS, verbose=1)

    rr_results = evaluate_policy(RoundRobinBaseline(), env_cls=EdgeCloudEnv)
    rl_results = evaluate_policy(make_policy(model), env_cls=EdgeCloudEnv)

    print_results("Baseline (round-robin)", rr_results)
    print_results("RL agent (PPO)", rl_results)


if __name__ == "__main__":
    main()
