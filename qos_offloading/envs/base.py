import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .. import config as C


class EdgeCloudEnv(gym.Env):
    critical_prob = C.CRITICAL_PROB_BASELINE

    def __init__(self, max_steps=C.MAX_STEPS):
        super().__init__()

        self.max_steps = max_steps
        self.num_edge_nodes = C.NUM_EDGE_NODES
        self.edge_capacity = C.EDGE_CAPACITY
        self.cloud_extra_delay = C.CLOUD_EXTRA_DELAY_MS
        self.sla_thresholds = {C.CRITICAL: C.CRITICAL_SLA_MS,
                               C.ANALYTICS: C.ANALYTICS_SLA_MS}
        self.violation_penalty = dict(C.VIOLATION_PENALTY)
        self.base_processing_time = C.BASE_PROCESSING_TIME_MS

        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([self.edge_capacity, self.edge_capacity, 1, 1], dtype=np.float32),
        )
        self.action_space = spaces.Discrete(C.NUM_ACTIONS)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.edge_queues = [0] * self.num_edge_nodes
        self.current_step = 0
        self.current_task_class = self._sample_task_class()
        return self._get_obs(), {}

    def step(self, action):
        task_class = self.current_task_class
        sla = self.sla_thresholds[task_class]

        latency = self._route(action)
        self._drain_queues()

        violated = latency > sla
        reward = self._reward(action, task_class, violated)

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        self.current_task_class = self._sample_task_class()

        info = {"latency": latency, "violated": violated, "task_class": task_class}
        return self._get_obs(), reward, terminated, False, info

    def _sample_task_class(self):
        return C.CRITICAL if self.np_random.random() < self.critical_prob else C.ANALYTICS

    def _route(self, action):
        if action in (C.EDGE_1, C.EDGE_2):
            queue_len = self.edge_queues[action]
            latency = self.base_processing_time + queue_len * C.CONGESTION_PENALTY_PER_TASK_MS
            self.edge_queues[action] = min(queue_len + 1, self._queue_cap())
        else:
            latency = self.base_processing_time + self.cloud_extra_delay
        return latency

    def _queue_cap(self):
        return self.edge_capacity

    def _drain_queues(self):
        self.edge_queues = [max(0, q - 1) for q in self.edge_queues]

    def _reward(self, action, task_class, violated):
        reward = -self.violation_penalty[task_class] if violated else C.SUCCESS_REWARD
        if action == C.CLOUD:
            reward -= C.CLOUD_USAGE_PENALTY
        return reward

    def _get_obs(self):
        steps_remaining_frac = 1.0 - (self.current_step / self.max_steps)
        return np.array([
            self.edge_queues[0],
            self.edge_queues[1],
            self.current_task_class,
            steps_remaining_frac,
        ], dtype=np.float32)
