from .. import config as C
from .base import EdgeCloudEnv
from .class_blind import ClassBlindRewardMixin


class EdgeCloudEnvHighLoad(EdgeCloudEnv):
    critical_prob = C.CRITICAL_PROB_HIGH_LOAD
    drain_prob = C.DRAIN_PROB_HIGH_LOAD

    def _drain_queues(self):
        self.edge_queues = [
            max(0, q - (1 if self.np_random.random() < self.drain_prob else 0))
            for q in self.edge_queues
        ]


class EdgeCloudEnvHighLoadClassBlind(ClassBlindRewardMixin, EdgeCloudEnvHighLoad):
    flat_penalty = C.FLAT_PENALTY_BASELINE
