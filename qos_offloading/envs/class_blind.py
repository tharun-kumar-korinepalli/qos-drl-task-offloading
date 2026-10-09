from .. import config as C
from .base import EdgeCloudEnv


class ClassBlindRewardMixin:
    flat_penalty = C.FLAT_PENALTY_BASELINE

    def _reward(self, action, task_class, violated):
        reward = -self.flat_penalty if violated else C.SUCCESS_REWARD
        if action == C.CLOUD:
            reward -= C.CLOUD_USAGE_PENALTY
        return reward


class EdgeCloudEnvClassBlind(ClassBlindRewardMixin, EdgeCloudEnv):
    flat_penalty = C.FLAT_PENALTY_BASELINE
