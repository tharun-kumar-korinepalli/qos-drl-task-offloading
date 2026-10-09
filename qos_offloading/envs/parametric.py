from .. import config as C
from ..params import EnvParams
from .scarce import EdgeCloudEnvScarce


class OffloadingEnv(EdgeCloudEnvScarce):
    def __init__(self, params=EnvParams()):
        self.params = params
        super().__init__(max_steps=params.max_steps, analytics_sla=params.analytics_sla,
                         critical_prob=params.critical_prob, drain_prob=params.drain_prob,
                         arrival_rate=params.arrival_rate)
        self.sla_thresholds = {C.CRITICAL: params.critical_sla, C.ANALYTICS: params.analytics_sla}
        cp, ap = params.penalties()
        self.violation_penalty = {C.CRITICAL: float(cp), C.ANALYTICS: float(ap)}

    def _reward(self, action, task_class, violated):
        return self.params.reward_scale * super()._reward(action, task_class, violated)
