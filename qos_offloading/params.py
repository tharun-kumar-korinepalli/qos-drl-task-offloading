from dataclasses import asdict, dataclass, replace

from . import config as C


@dataclass(frozen=True)
class EnvParams:
    critical_sla: float = C.CRITICAL_SLA_MS
    analytics_sla: float = C.ANALYTICS_SLA_SCARCE_MS
    critical_prob: float = C.CRITICAL_PROB_SCARCE
    drain_prob: float = C.DRAIN_PROB_SCARCE
    arrival_rate: float = C.ARRIVAL_RATE_SCARCE
    critical_penalty: float = C.VIOLATION_PENALTY[C.CRITICAL]
    analytics_penalty: float = C.VIOLATION_PENALTY[C.ANALYTICS]
    flat: bool = False
    max_steps: int = C.MAX_STEPS
    reward_scale: float = 1.0

    def matched_flat_penalty(self):
        p = self.critical_prob
        return p * self.critical_penalty + (1.0 - p) * self.analytics_penalty

    def penalties(self):
        if self.flat:
            f = self.matched_flat_penalty()
            return f, f
        return self.critical_penalty, self.analytics_penalty

    def canonical(self):
        cp, ap = self.penalties()
        return replace(self, critical_penalty=float(cp), analytics_penalty=float(ap), flat=False)

    def eval_view(self):
        return replace(self, critical_penalty=C.EVAL_PENALTY[C.CRITICAL],
                       analytics_penalty=C.EVAL_PENALTY[C.ANALYTICS], flat=False, reward_scale=1.0)

    def utilization(self):
        return self.arrival_rate / (C.NUM_EDGE_NODES * self.drain_prob)

    def to_dict(self):
        d = asdict(self)
        if d["reward_scale"] == 1.0:
            d.pop("reward_scale")
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


WEIGHTED = EnvParams()
FLAT = EnvParams(flat=True)
