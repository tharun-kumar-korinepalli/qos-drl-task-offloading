import numpy as np
import pytest

from qos_offloading import config as C
from qos_offloading.envs import (
    EdgeCloudEnv, EdgeCloudEnvClassBlind, EdgeCloudEnvHighLoad,
    EdgeCloudEnvHighLoadClassBlind, EdgeCloudEnvScarce, EdgeCloudEnvScarceClassBlind,
)
from qos_offloading.baselines import RoundRobinBaseline
from qos_offloading.evaluation import evaluate_policy, summarize

ALL_ENVS = [EdgeCloudEnv, EdgeCloudEnvClassBlind, EdgeCloudEnvHighLoad,
            EdgeCloudEnvHighLoadClassBlind, EdgeCloudEnvScarce, EdgeCloudEnvScarceClassBlind]


@pytest.mark.parametrize("env_cls", ALL_ENVS)
def test_episode_runs_to_completion(env_cls):
    env = env_cls()
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    steps = 0
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert set(info) == {"latency", "violated", "task_class"}
        done = terminated or truncated
        steps += 1
    assert steps == C.MAX_STEPS


@pytest.mark.parametrize("env_cls", ALL_ENVS)
def test_reset_is_reproducible(env_cls):
    def rollout():
        env = env_cls()
        env.reset(seed=123)
        return [env.step(i % 3)[1] for i in range(50)]
    assert rollout() == rollout()


def test_cloud_latency_is_constant():
    env = EdgeCloudEnv()
    env.reset(seed=0)
    for _ in range(10):
        _, _, _, _, info = env.step(C.CLOUD)
        assert info["latency"] == C.BASE_PROCESSING_TIME_MS + C.CLOUD_EXTRA_DELAY_MS


def test_class_blind_reward_is_flat():
    env = EdgeCloudEnvScarceClassBlind()
    env.reset(seed=0)
    rewards = {env.step(C.CLOUD)[1] for _ in range(20)}
    assert rewards == {-C.FLAT_PENALTY_SCARCE - C.CLOUD_USAGE_PENALTY}


def test_evaluate_policy_returns_expected_keys():
    r = evaluate_policy(RoundRobinBaseline(), env_cls=EdgeCloudEnv, n_episodes=2)
    assert set(r) == {"class_0_violation_rate", "class_0_avg_latency",
                      "class_1_violation_rate", "class_1_avg_latency",
                      "overall_violation_rate", "sla_cost_per_task", "edge_utilization"}
    assert 0.0 <= r["edge_utilization"] <= 1.0


def test_summarize():
    s = summarize([{"a": 1.0}, {"a": 3.0}])
    assert s["a"] == (2.0, 1.0)
