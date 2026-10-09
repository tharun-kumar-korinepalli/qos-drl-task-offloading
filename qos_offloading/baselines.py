from . import config as C


class RoundRobinBaseline:
    def __init__(self):
        self.i = 0

    def __call__(self, obs):
        action = self.i % C.NUM_ACTIONS
        self.i += 1
        return action


class ShortestQueueBaseline:
    def __call__(self, obs):
        return C.EDGE_1 if obs[0] <= obs[1] else C.EDGE_2


class SLAThresholdBaseline:
    def __init__(self, sla_thresholds):
        self.sla_thresholds = sla_thresholds

    def __call__(self, obs):
        edge = C.EDGE_1 if obs[0] <= obs[1] else C.EDGE_2
        predicted = C.BASE_PROCESSING_TIME_MS + obs[edge] * C.CONGESTION_PENALTY_PER_TASK_MS
        if predicted <= self.sla_thresholds[int(obs[2])]:
            return edge
        return C.CLOUD


class ReservationBaseline:
    def __init__(self, sla_thresholds, analytics_queue_limit):
        self.sla_thresholds = sla_thresholds
        self.limit = analytics_queue_limit

    def __call__(self, obs):
        edge = C.EDGE_1 if obs[0] <= obs[1] else C.EDGE_2
        q = obs[edge]
        predicted = C.BASE_PROCESSING_TIME_MS + q * C.CONGESTION_PENALTY_PER_TASK_MS
        if predicted > self.sla_thresholds[int(obs[2])]:
            return C.CLOUD
        if int(obs[2]) == C.ANALYTICS and q > self.limit:
            return C.CLOUD
        return edge


class TablePolicy:
    def __init__(self, table, max_steps=C.MAX_STEPS, queue_cap=C.SCARCE_QUEUE_CAP):
        self.table = table
        self.T = max_steps
        self.Q = queue_cap + 1

    def __call__(self, obs):
        t = int(round((1.0 - float(obs[3])) * self.T))
        s = int(obs[0]) * self.Q + int(obs[1])
        return int(self.table[min(t, self.T - 1), s, int(obs[2])])
