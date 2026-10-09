import numpy as np
from gymnasium import spaces

from .. import config as C
from .base import EdgeCloudEnv
from .class_blind import ClassBlindRewardMixin


class EdgeCloudEnvScarce(EdgeCloudEnv):
    critical_prob = C.CRITICAL_PROB_SCARCE

    def __init__(self, max_steps=C.MAX_STEPS,
                 analytics_sla=C.ANALYTICS_SLA_SCARCE_MS,
                 critical_prob=C.CRITICAL_PROB_SCARCE,
                 drain_prob=C.DRAIN_PROB_SCARCE,
                 arrival_rate=C.ARRIVAL_RATE_SCARCE):
        self.analytics_sla = analytics_sla
        self.critical_prob = critical_prob
        self.drain_prob = drain_prob
        self.arrival_rate = arrival_rate
        self.pending_extra = 0.0
        super().__init__(max_steps=max_steps)

        self.sla_thresholds = {C.CRITICAL: C.CRITICAL_SLA_MS, C.ANALYTICS: analytics_sla}
        self.edge_cap_soft = self.edge_capacity + C.SCARCE_EXTRA_CAPACITY
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([self.edge_cap_soft, self.edge_cap_soft, 1, 1], dtype=np.float32),
        )

    def reset(self, seed=None, options=None):
        self.pending_extra = 0.0
        return super().reset(seed=seed, options=options)

    def _queue_cap(self):
        return self.edge_cap_soft

    def _drain_queues(self):
        self.pending_extra += (self.arrival_rate - 1.0)
        while self.pending_extra >= 1.0:
            target = int(np.argmin(self.edge_queues))
            self.edge_queues[target] = min(self.edge_queues[target] + 1, self.edge_cap_soft)
            self.pending_extra -= 1.0

        self.edge_queues = [
            max(0, q - (1 if self.np_random.random() < self.drain_prob else 0))
            for q in self.edge_queues
        ]


class EdgeCloudEnvScarceClassBlind(ClassBlindRewardMixin, EdgeCloudEnvScarce):
    flat_penalty = C.FLAT_PENALTY_SCARCE


def make_scarce_env_with_penalty(critical_penalty, analytics_penalty):
    class EdgeCloudEnvScarcePenalty(EdgeCloudEnvScarce):
        def __init__(self, max_steps=C.MAX_STEPS):
            super().__init__(max_steps=max_steps)
            self.violation_penalty = {C.CRITICAL: float(critical_penalty),
                                      C.ANALYTICS: float(analytics_penalty)}

    EdgeCloudEnvScarcePenalty.__name__ = f"EdgeCloudEnvScarcePenalty_{critical_penalty}_{analytics_penalty}"
    return EdgeCloudEnvScarcePenalty
