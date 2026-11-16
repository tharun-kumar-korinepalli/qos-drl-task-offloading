from itertools import product

import numpy as np

from .exact import ExactModel


class Rules:
    def __init__(self, system):
        self.sp = system
        self.base = np.array(system.base_ms)
        self.slope = np.array(system.slope_ms)
        self.lim = np.array(system.limits_ms)
        self.N = system.n

    def jsq(self, q, c):
        return int(np.argmin(q))

    def best_edge(self, q):
        pred = self.base + self.slope * np.asarray(q)
        e = int(np.argmin(pred))
        return e, pred[e]

    def sla(self, q, c):
        e, pr = self.best_edge(q)
        return e if pr <= self.lim[c] else self.N

    def reservation(self, thresholds):
        th = np.array(thresholds, dtype=float)

        def rule(q, c):
            e, pr = self.best_edge(q)
            return e if pr <= min(self.lim[c], th[c]) else self.N
        return rule

    def node_sets(self, thresholds, allowed):
        th = np.array(thresholds, dtype=float)
        masks = [np.array([i in a for i in range(self.N)]) for a in allowed]

        def rule(q, c):
            pred = np.where(masks[c], self.base + self.slope * np.asarray(q), np.inf)
            e = int(np.argmin(pred))
            return e if pred[e] <= min(self.lim[c], th[c]) else self.N
        return rule


def _latencies(sp):
    return sorted(set((np.array(sp.base_ms)[None, :] + np.array(sp.slope_ms)[None, :]
                       * np.arange(sp.cap + 1)[:, None]).ravel().tolist()))


def tune_reservation(model, rules, max_passes=4):
    sp = rules.sp
    vals = _latencies(sp)
    cand = [[-1.0] + [v for v in vals if v <= sp.limits_ms[c]] for c in range(1, sp.k)]

    def ev(th):
        return model.evaluate(model.policy_from_rule(rules.reservation((float("inf"),) + tuple(th))))

    if sp.k == 2:
        m, th = min(((ev((x,)), (x,)) for x in cand[0]), key=lambda r: r[0]["cost"])
        return (float("inf"),) + th, m
    th = [c[-1] for c in cand]
    best = ev(th)
    for _ in range(max_passes):
        changed = False
        for i, opts in enumerate(cand):
            for x in opts:
                if x == th[i]:
                    continue
                trial = list(th)
                trial[i] = x
                m = ev(trial)
                if m["cost"] < best["cost"] - 1e-12:
                    best, th, changed = m, trial, True
        if not changed:
            break
    return (float("inf"),) + tuple(th), best


def tune_node_sets(model, rules, start, max_passes=4):
    sp = rules.sp
    vals = _latencies(sp)
    subsets = [tuple(i for i in range(sp.n) if (m >> i) & 1) for m in range(1, 2 ** sp.n)]
    th = list(start)
    allowed = [tuple(range(sp.n))] * sp.k

    def ev(th, allowed):
        return model.evaluate(model.policy_from_rule(rules.node_sets(th, allowed)))

    best = ev(th, allowed)
    for _ in range(max_passes):
        changed = False
        for c in range(sp.k):
            ths = [float("inf")] if c == 0 else [-1.0] + [v for v in vals if v <= sp.limits_ms[c]]
            for sub, x in product(subsets, ths):
                if sub == allowed[c] and x == th[c]:
                    continue
                t2, a2 = list(th), list(allowed)
                t2[c], a2[c] = x, sub
                m = ev(t2, a2)
                if m["cost"] < best["cost"] - 1e-12:
                    best, th, allowed, changed = m, t2, a2, True
        if not changed:
            break
    return th, allowed, best


def _finite(th):
    return [None if not np.isfinite(x) else x for x in th]


def reference(system, node_sets=False):
    model = ExactModel(system)
    rules = Rules(system)
    out = {"load": system.load(), "states": model.S * model.K * model.T}
    out["jsq"] = model.evaluate(model.policy_from_rule(rules.jsq))
    out["sla_rule"] = model.evaluate(model.policy_from_rule(rules.sla))
    th, m = tune_reservation(model, rules)
    out["reservation"] = {**m, "thresholds": _finite(th)}
    if node_sets:
        th2, allowed, m2 = tune_node_sets(model, rules, th)
        out["node_sets"] = {**m2, "thresholds": _finite(th2), "allowed": [list(a) for a in allowed]}
    out["optimum"] = model.evaluate(model.solve(model.cost_eval())[0])
    return model, out, th
