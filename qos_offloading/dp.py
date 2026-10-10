import numpy as np

from . import config as C
from .params import EnvParams


def background_schedule(params):
    pending = 0.0
    sched = []
    for _ in range(params.max_steps):
        pending += (params.arrival_rate - 1.0)
        k = 0
        while pending >= 1.0:
            k += 1
            pending -= 1.0
        sched.append(k)
    return np.array(sched, dtype=int)


class ExactModel:
    def __init__(self, params=EnvParams()):
        self.params = params
        self.T = params.max_steps
        self.cap = C.SCARCE_QUEUE_CAP
        self.Q = self.cap + 1
        self.S = self.Q * self.Q
        self.p_crit = params.critical_prob
        self.class_prob = np.array([self.p_crit, 1.0 - self.p_crit])
        self.sla = np.array([params.critical_sla, params.analytics_sla])
        self.sched = background_schedule(params)
        q1, q2 = np.divmod(np.arange(self.S), self.Q)
        self.q1, self.q2 = q1, q2
        self.latency = np.zeros((self.S, C.NUM_ACTIONS))
        self.latency[:, C.EDGE_1] = C.BASE_PROCESSING_TIME_MS + q1 * C.CONGESTION_PENALTY_PER_TASK_MS
        self.latency[:, C.EDGE_2] = C.BASE_PROCESSING_TIME_MS + q2 * C.CONGESTION_PENALTY_PER_TASK_MS
        self.latency[:, C.CLOUD] = C.BASE_PROCESSING_TIME_MS + C.CLOUD_EXTRA_DELAY_MS
        self.violated = self.latency[:, :, None] > self.sla[None, None, :]
        self.P = {k: self._transition(k) for k in sorted(set(self.sched.tolist()))}

    def _idx(self, a, b):
        return a * self.Q + b

    def _transition(self, k):
        d = self.params.drain_prob
        P = np.zeros((C.NUM_ACTIONS, self.S, self.S))
        for s in range(self.S):
            for a in range(C.NUM_ACTIONS):
                q = [int(self.q1[s]), int(self.q2[s])]
                if a in (C.EDGE_1, C.EDGE_2):
                    q[a] = min(q[a] + 1, self.cap)
                for _ in range(k):
                    j = int(np.argmin(q))
                    q[j] = min(q[j] + 1, self.cap)
                for x1, p1 in ((q[0], 1.0),) if q[0] == 0 else ((q[0] - 1, d), (q[0], 1.0 - d)):
                    for x2, p2 in ((q[1], 1.0),) if q[1] == 0 else ((q[1] - 1, d), (q[1], 1.0 - d)):
                        P[a, s, self._idx(x1, x2)] += p1 * p2
        return P

    def step_cost(self, crit_cost, ana_cost, success_reward=0.0, cloud_cost=0.0):
        cost = np.where(self.violated, np.array([crit_cost, ana_cost])[None, None, :], -success_reward)
        cost = cost.astype(float)
        cost[:, C.CLOUD, :] += cloud_cost
        return cost

    def solve(self, cost, gamma=1.0):
        T, S = self.T, self.S
        policy = np.zeros((T, S, 2), dtype=int)
        V = np.zeros((S, 2))
        for t in range(T - 1, -1, -1):
            W = V @ self.class_prob
            Pt = self.P[self.sched[t]]
            Qv = cost + gamma * np.stack([Pt[a] @ W for a in range(C.NUM_ACTIONS)], axis=1)[:, :, None]
            policy[t] = np.argmin(Qv, axis=1)
            V = np.min(Qv, axis=1)
        return policy, float(V[0] @ self.class_prob)

    def evaluate(self, policy):
        T, S = self.T, self.S
        dist = np.zeros((S, 2))
        dist[0] = self.class_prob
        acc = dict(viol=np.zeros(2), count=np.zeros(2), lat=np.zeros(2), edge=0.0)
        rows = np.arange(S)
        for t in range(T):
            act = policy[t]
            nxt = np.zeros(S)
            Pt = self.P[self.sched[t]]
            for c in range(2):
                a = act[:, c]
                w = dist[:, c]
                acc["count"][c] += w.sum()
                acc["viol"][c] += w @ self.violated[rows, a, c]
                acc["lat"][c] += w @ self.latency[rows, a]
                acc["edge"] += w @ (a != C.CLOUD)
                for u in range(C.NUM_ACTIONS):
                    m = a == u
                    if m.any():
                        nxt += w[m] @ Pt[u][m]
            dist = nxt[:, None] * self.class_prob[None, :]
        return metrics_from_counts(acc, T, self.p_crit)

    def all_observations(self):
        t = np.repeat(np.arange(self.T), self.S * 2)
        s = np.tile(np.repeat(np.arange(self.S), 2), self.T)
        c = np.tile(np.arange(2), self.T * self.S)
        frac = np.array([np.float32(1.0 - (ti / self.T)) for ti in range(self.T)], dtype=np.float32)
        obs = np.stack([self.q1[s], self.q2[s], c, frac[t]], axis=1).astype(np.float32)
        return obs

    def policy_from_fn(self, fn):
        obs = self.all_observations()
        acts = np.asarray(fn(obs)).astype(int).reshape(self.T, self.S, 2)
        return acts

    def policy_from_obs_rule(self, rule):
        obs = self.all_observations()
        acts = np.array([rule(o) for o in obs], dtype=int).reshape(self.T, self.S, 2)
        return acts

    def round_robin_policies(self):
        out = []
        for phase in range(C.NUM_ACTIONS):
            pol = np.zeros((self.T, self.S, 2), dtype=int)
            for t in range(self.T):
                pol[t] = (phase + t) % C.NUM_ACTIONS
            out.append(pol)
        return out

    def evaluate_mixture(self, policies):
        ms = [self.evaluate(p) for p in policies]
        keys = ms[0]["_raw"].keys()
        raw = {k: np.mean([m["_raw"][k] for m in ms], axis=0) for k in keys}
        return metrics_from_counts(raw, self.T, self.p_crit)


def metrics_from_counts(acc, T, p_crit):
    viol, count, lat = (np.asarray(acc[k], dtype=float) for k in ("viol", "count", "lat"))
    vr = viol / count
    out = {
        "class_0_violation_rate": float(vr[0]),
        "class_0_avg_latency": float(lat[0] / count[0]),
        "class_1_violation_rate": float(vr[1]),
        "class_1_avg_latency": float(lat[1] / count[1]),
        "overall_violation_rate": float(viol.sum() / count.sum()),
        "sla_cost_per_task": float(p_crit * C.EVAL_PENALTY[C.CRITICAL] * vr[0]
                                   + (1.0 - p_crit) * C.EVAL_PENALTY[C.ANALYTICS] * vr[1]),
        "edge_utilization": float(acc["edge"] / count.sum()),
    }
    out["_raw"] = {"viol": viol.tolist(), "count": count.tolist(), "lat": lat.tolist(),
                   "edge": float(acc["edge"])}
    return out


def strip(m):
    return {k: v for k, v in m.items() if not k.startswith("_")}


def optimal_sla_policy(model):
    cost = model.step_cost(C.EVAL_PENALTY[C.CRITICAL], C.EVAL_PENALTY[C.ANALYTICS])
    return model.solve(cost)[0]


def optimal_reward_policy(model, params):
    cp, ap = params.penalties()
    cost = model.step_cost(cp, ap, success_reward=C.SUCCESS_REWARD, cloud_cost=C.CLOUD_USAGE_PENALTY)
    return model.solve(cost)[0]


def pareto_frontier(model, n=101):
    pts = []
    for lam in np.linspace(0.0, 1.0, n):
        cost = model.step_cost(lam / model.p_crit if model.p_crit > 0 else 0.0,
                               (1.0 - lam) / (1.0 - model.p_crit) if model.p_crit < 1 else 0.0)
        pol, _ = model.solve(cost)
        m = strip(model.evaluate(pol))
        pts.append((float(lam), m["class_0_violation_rate"], m["class_1_violation_rate"]))
    return pts
