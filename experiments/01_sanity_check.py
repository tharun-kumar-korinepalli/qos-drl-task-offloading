from stable_baselines3.common.env_checker import check_env

from qos_offloading.envs import (
    EdgeCloudEnv, EdgeCloudEnvClassBlind, EdgeCloudEnvHighLoad,
    EdgeCloudEnvHighLoadClassBlind, EdgeCloudEnvScarce, EdgeCloudEnvScarceClassBlind, OffloadingEnv,
)
from qos_offloading.params import FLAT, WEIGHTED


def main():
    env = EdgeCloudEnv()
    obs, info = env.reset()
    print("Initial observation:", obs)
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"step {i}: action={action}, reward={reward:.2f}, info={info}")

    for cls in (EdgeCloudEnv, EdgeCloudEnvClassBlind, EdgeCloudEnvHighLoad,
                EdgeCloudEnvHighLoadClassBlind, EdgeCloudEnvScarce, EdgeCloudEnvScarceClassBlind):
        check_env(cls(), warn=True)
        print(f"{cls.__name__}: environment check passed")
    for name, p in [("weighted", WEIGHTED), ("flat", FLAT)]:
        check_env(OffloadingEnv(p), warn=True)
        print(f"OffloadingEnv ({name}): environment check passed")


if __name__ == "__main__":
    main()
