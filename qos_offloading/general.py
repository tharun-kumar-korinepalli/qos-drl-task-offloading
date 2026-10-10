"""General offloading system: N heterogeneous edge nodes, K task classes, a cloud that may meet some
limits, deterministic or random background traffic, and an exact finite-horizon model of it.

With the default parameters the system is the same as OffloadingEnv (see tests/test_general.py).
"""
from dataclasses import asdict, dataclass, field, replace
from itertools import product

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy import sparse


@dataclass(frozen=True)
class SysParams:
    base_ms: tuple = (30.0, 30.0)
    slope_ms: tuple = (15.0, 15.0)
    drain: tuple = (0.6, 0.6)
    cap: int = 8
    class_probs: tuple = (0.5, 0.5)
    limits_ms: tuple = (50.0, 90.0)
    eval_weights: tuple = (10.0, 2.0)
    cloud_ms: float = 110.0
    cloud_meet: tuple = (0.0, 0.0)
    background: float = 0.2
    background_mode: str = "deterministic"
    max_steps: int = 200

    @property
    def n(self):
        return len(self.base_ms)

    @property
    def k(self):
        return len(self.class_probs)

    def load(self):
        return (1.0 + self.background) / float(sum(self.drain))

    def to_dict(self):
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()})


@dataclass(frozen=True)
class RewardSpec:
    kind: str = "weighted"
    penalties: tuple = (10.0, 2.0)
    success: float = 1.0
    cloud_cost: float = 0.1
    scale: float = 1.0
    target: tuple = ()
    eta: float = 0.0
    window: int = 200

    def effective_penalties(self, sysp):
        if self.kind == "flat":
            m = float(np.dot(sysp.class_probs, self.penalties))
            return tuple([m] * sysp.k)
        return tuple(float(x) for x in self.penalties)

    def to_dict(self):
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()})


def corrected_penalties(ratios, probs, mean_penalty, success=1.0):
    """Penalties w with (w_k + success) proportional to the operator weights and sum_k p_k w_k = mean."""
    r = np.asarray(ratios, dtype=float)
    p = np.asarray(probs, dtype=float)
    c = (mean_penalty + success) / float(p @ r)
    return tuple(float(x) for x in c * r - success)


def background_counts(sysp, rng=None):
    """Deterministic per-step background arrivals (deterministic mode)."""
    pending = 0.0
    out = []
    for _ in range(sysp.max_steps):
        pending += (1.0 + sysp.background) - 1.0  # same float steps as the original simulator
        k = 0
        while pending >= 1.0:
            k += 1
            pending -= 1.0
        out.append(k)
    return np.array(out, dtype=int)


def background_dist(sysp):
    """Distribution of background arrivals per step in random mode: floor(b) plus Bernoulli(frac(b))."""
    lo = int(np.floor(sysp.background))
    frac = sysp.background - lo
    if frac < 1e-12:
        return {lo: 1.0}
    return {lo: 1.0 - frac, lo + 1: frac}


class GeneralEnv(gym.Env):
    """Observation: queue lengths of the N edge nodes, class index of the task, share of episode left."""

    def __init__(self, sysp=SysParams(), reward=RewardSpec(), svrs_features=False, prune=False):
        super().__init__()
        self.sp = sysp
        self.rw = reward
        self.svrs = svrs_features
        self.prune = prune
        self.N, self.K = sysp.n, sysp.k
        self.base = np.array(sysp.base_ms)
        self.slope = np.array(sysp.slope_ms)
        self.limits = np.array(sysp.limits_ms)
        self.pen = np.array(reward.effective_penalties(sysp))
        self.lam = self.pen.copy()
        self.hist = [[] for _ in range(self.K)]
        hi = [sysp.cap] * self.N + [self.K - 1, 1]
        lo = [0] * (self.N + 2)
        if self.svrs:
            hi += [4.0] * self.N + [1.0] * self.K
            lo += [0.0] * (self.N + self.K)
        self.observation_space = spaces.Box(low=np.array(lo, dtype=np.float32), high=np.array(hi, dtype=np.float32))
        self.action_space = spaces.Discrete(self.N + 1)
        self.sched = background_counts(sysp) if sysp.background_mode == "deterministic" else None
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.q = np.zeros(self.N, dtype=int)
        self.t = 0
        self.pending = 0.0
        self.c = self._sample_class()
        return self._obs(), {}

    def _sample_class(self):
        u = self.np_random.random()
        return int(min(np.searchsorted(np.cumsum(self.sp.class_probs), u, side="right"), self.K - 1))

    def predicted(self):
        return self.base + self.slope * self.q

    def action_masks(self):
        m = np.ones(self.N + 1, dtype=bool)
        if self.prune:
            m[: self.N] = self.predicted() <= self.limits[self.c]
        return m

    def _obs(self):
        o = list(self.q.astype(float)) + [float(self.c), float(np.float32(1.0 - self.t / self.sp.max_steps))]
        if self.svrs:
            o += list(np.minimum(self.predicted() / self.limits[self.c], 4.0))
            o += [float(np.mean(h)) if h else 0.0 for h in self.hist]
        return np.array(o, dtype=np.float32)

    def step(self, action):
        a = int(action)
        c = self.c
        if a < self.N:
            lat = self.base[a] + self.slope[a] * self.q[a]
            miss = bool(lat > self.limits[c])
            self.q[a] = min(self.q[a] + 1, self.sp.cap)
        else:
            lat = self.sp.cloud_ms
            pm = self.sp.cloud_meet[c]
            miss = True if pm <= 0.0 else (False if pm >= 1.0 else bool(self.np_random.random() >= pm))
        self._background()
        for i in range(self.N):
            if self.np_random.random() < self.sp.drain[i]:
                self.q[i] = max(0, self.q[i] - 1)
        r = self._reward(a, c, miss)
        self.t += 1
        done = self.t >= self.sp.max_steps
        self.c = self._sample_class()
        info = {"latency": float(lat), "violated": miss, "task_class": c}
        return self._obs(), r, done, False, info

    def _background(self):
        if self.sched is not None:
            k = int(self.sched[self.t])
        else:
            lo = int(np.floor(self.sp.background))
            frac = self.sp.background - lo
            k = lo + (1 if frac > 0 and self.np_random.random() < frac else 0)
        for _ in range(k):
            j = int(np.argmin(self.q))
            self.q[j] = min(self.q[j] + 1, self.sp.cap)

    def _reward(self, a, c, miss):
        rw = self.rw
        if rw.kind == "tiered_adaptive":
            pen = self.lam[c]
            self.hist[c].append(1.0 if miss else 0.0)
            if len(self.hist[c]) > rw.window:
                self.hist[c].pop(0)
            rate = float(np.mean(self.hist[c]))
            self.lam[c] = max(0.0, self.lam[c] + rw.eta * (rate - rw.target[c]))
        else:
            pen = self.pen[c]
            if self.svrs:
                self.hist[c].append(1.0 if miss else 0.0)
                if len(self.hist[c]) > rw.window:
                    self.hist[c].pop(0)
        r = -pen if miss else rw.success
        if a == self.N:
            r -= rw.cloud_cost
        return rw.scale * r


class GeneralModel:
    """Exact finite-horizon model of GeneralEnv with sparse transition matrices."""

    def __init__(self, sysp=SysParams()):
        self.sp = sysp
        self.N, self.K, self.T = sysp.n, sysp.k, sysp.max_steps
        self.Q = sysp.cap + 1
        self.S = self.Q ** self.N
        self.A = self.N + 1
        self.p = np.array(sysp.class_probs, dtype=float)
        grids = np.indices([self.Q] * self.N).reshape(self.N, -1).T
        self.qs = grids
        self.radix = self.Q ** np.arange(self.N - 1, -1, -1)
        lat = np.zeros((self.S, self.A))
        for i in range(self.N):
            lat[:, i] = sysp.base_ms[i] + sysp.slope_ms[i] * self.qs[:, i]
        lat[:, self.N] = sysp.cloud_ms
        self.latency = lat
        lim = np.array(sysp.limits_ms)
        miss = (lat[:, :, None] > lim[None, None, :]).astype(float)
        miss[:, self.N, :] = 1.0 - np.array(sysp.cloud_meet)[None, :]
        self.miss = miss
        if sysp.background_mode == "deterministic":
            self.sched = background_counts(sysp)
            ks = sorted(set(self.sched.tolist()))
            self.P = {k: self._build({k: 1.0}) for k in ks}
            self.P_at = lambda t: self.P[int(self.sched[t])]
        else:
            Pm = self._build(background_dist(sysp))
            self.P = {"r": Pm}
            self.P_at = lambda t: Pm
        self.PT = {key: sparse.hstack([m.T for m in mats]).tocsr() for key, mats in self.P.items()}

    def index(self, q):
        return np.asarray(q) @ self.radix

    def _build(self, kdist):
        S, N, A = self.S, self.N, self.A
        d = np.array(self.sp.drain)
        mats = []
        patterns = np.array(list(product([0, 1], repeat=N)), dtype=int)
        pprob = np.prod(np.where(patterns == 1, d[None, :], 1.0 - d[None, :]), axis=1)
        for a in range(A):
            rows, cols, vals = [], [], []
            for kb, pk in kdist.items():
                q = self.qs.copy()
                if a < N:
                    q[:, a] = np.minimum(q[:, a] + 1, self.sp.cap)
                for _ in range(kb):
                    j = np.argmin(q, axis=1)
                    q[np.arange(S), j] = np.minimum(q[np.arange(S), j] + 1, self.sp.cap)
                for pat, pp in zip(patterns, pprob):
                    nq = np.maximum(q - pat[None, :], 0)
                    rows.append(np.arange(S))
                    cols.append(nq @ self.radix)
                    vals.append(np.full(S, pk * pp))
            M = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(S, S))
            M.sum_duplicates()
            mats.append(M)
        return mats

    def _key(self, t):
        return int(self.sched[t]) if self.sp.background_mode == "deterministic" else "r"

    # costs -----------------------------------------------------------------------------------
    def cost_eval(self, weights=None):
        w = np.array(weights if weights is not None else self.sp.eval_weights, dtype=float)
        return self.miss * w[None, None, :]

    def cost_reward(self, reward):
        w = np.array(reward.effective_penalties(self.sp), dtype=float)
        c = self.miss * w[None, None, :] - reward.success * (1.0 - self.miss)
        c[:, self.N, :] += reward.cloud_cost
        return c

    # dynamic programming ---------------------------------------------------------------------
    def solve(self, cost, gamma=1.0):
        pol = np.zeros((self.T, self.S, self.K), dtype=np.int8)
        V = np.zeros((self.S, self.K))
        for t in range(self.T - 1, -1, -1):
            W = V @ self.p
            mats = self.P[self._key(t)]
            cont = np.stack([m @ W for m in mats], axis=1)
            Qv = cost + gamma * cont[:, :, None]
            pol[t] = np.argmin(Qv, axis=1)
            V = np.min(Qv, axis=1)
        return pol, float(V[0] @ self.p)

    def _forward(self, policy, per_step):
        """Propagate the state distribution under a policy table; per_step(t, dist, act) accumulates."""
        S, K, A = self.S, self.K, self.A
        dist = np.zeros((S, K))
        dist[0] = self.p
        rows = np.arange(S)
        for t in range(self.T):
            act = policy[t].astype(np.int64)
            per_step(dist, act)
            v = np.zeros(A * S)
            for c in range(K):
                np.add.at(v, act[:, c] * S + rows, dist[:, c])
            nxt = self.PT[self._key(t)] @ v
            dist = nxt[:, None] * self.p[None, :]

    def evaluate(self, policy):
        K = self.K
        rows = np.arange(self.S)
        acc = {"viol": np.zeros(K), "cnt": np.zeros(K), "lat": np.zeros(K), "edge": 0.0}

        def step(dist, act):
            for c in range(K):
                a = act[:, c]
                w = dist[:, c]
                acc["cnt"][c] += w.sum()
                acc["viol"][c] += w @ self.miss[rows, a, c]
                acc["lat"][c] += w @ self.latency[rows, a]
                acc["edge"] += w @ (a != self.N)
        self._forward(policy, step)
        vr = acc["viol"] / acc["cnt"]
        ew = np.array(self.sp.eval_weights)
        return {"miss": vr.tolist(), "latency": (acc["lat"] / acc["cnt"]).tolist(),
                "overall": float(acc["viol"].sum() / acc["cnt"].sum()),
                "cost": float(self.p @ (ew * vr)), "edge": float(acc["edge"] / acc["cnt"].sum())}

    def expected_return(self, policy, reward):
        c = self.cost_reward(reward)
        rows = np.arange(self.S)
        tot = [0.0]

        def step(dist, act):
            for k in range(self.K):
                tot[0] -= dist[:, k] @ c[rows, act[:, k], k]
        self._forward(policy, step)
        return float(tot[0])

    # policies from rules and networks --------------------------------------------------------
    def observations(self):
        T, S, K, N = self.T, self.S, self.K, self.N
        frac = np.array([np.float32(1.0 - t / T) for t in range(T)], dtype=np.float32)
        obs = np.zeros((T, S, K, N + 2), dtype=np.float32)
        obs[:, :, :, :N] = self.qs[None, :, None, :]
        obs[:, :, :, N] = np.arange(K)[None, None, :]
        obs[:, :, :, N + 1] = frac[:, None, None]
        return obs.reshape(-1, N + 2)

    def policy_from_fn(self, fn, batch=200_000):
        obs = self.observations()
        out = np.concatenate([np.asarray(fn(obs[i:i + batch])).reshape(-1) for i in range(0, len(obs), batch)])
        return out.astype(np.int8).reshape(self.T, self.S, self.K)

    def policy_from_state_rule(self, rule):
        acts = np.zeros((self.S, self.K), dtype=np.int8)
        for s in range(self.S):
            for c in range(self.K):
                acts[s, c] = rule(self.qs[s], c)
        return np.broadcast_to(acts, (self.T, self.S, self.K)).copy()


def eval_weights_for(sysp, ratios):
    return tuple(float(x) for x in ratios)


# rules --------------------------------------------------------------------------------------------
class Rules:
    """Rule baselines. `model` may differ from the true system (model shift)."""

    def __init__(self, sysp):
        self.sp = sysp
        self.base = np.array(sysp.base_ms)
        self.slope = np.array(sysp.slope_ms)
        self.lim = np.array(sysp.limits_ms)
        self.N = sysp.n

    def jsq(self, q, c):
        return int(np.argmin(q))

    def best_edge(self, q):
        pred = self.base + self.slope * np.asarray(q)
        e = int(np.argmin(pred))
        return e, pred[e]

    def sla(self, q, c):
        e, pr = self.best_edge(q)
        return e if pr <= self.lim[c] else self.N

    def reservation_sets(self, thresholds, allowed):
        """Class c may use only the edge nodes in allowed[c], and only if the predicted latency of the best
        allowed node is at most thresholds[c]."""
        th = np.array(thresholds, dtype=float)
        masks = [np.array([i in a for i in range(self.N)]) for a in allowed]

        def rule(q, c):
            pred = np.where(masks[c], self.base + self.slope * np.asarray(q), np.inf)
            e = int(np.argmin(pred))
            return e if pred[e] <= min(self.lim[c], th[c]) else self.N
        return rule

    def reservation(self, thresholds):
        th = np.array(thresholds, dtype=float)

        def rule(q, c):
            e, pr = self.best_edge(q)
            return e if pr <= min(self.lim[c], th[c]) else self.N
        return rule


def _lat_values(sp):
    return sorted(set((np.array(sp.base_ms)[None, :] + np.array(sp.slope_ms)[None, :]
                       * np.arange(sp.cap + 1)[:, None]).ravel().tolist()))


def tune_reservation(model, rules, max_passes=4):
    """Per-class latency thresholds (class 0 unrestricted), chosen on the exact model.

    One restricted class: exhaustive search. More classes: coordinate descent started from the SLA rule,
    repeated until no threshold changes."""
    sp = rules.sp
    vals = _lat_values(sp)
    cand = [[-1.0] + [v for v in vals if v <= sp.limits_ms[c]] for c in range(1, sp.k)]
    ev = lambda th: model.evaluate(model.policy_from_state_rule(rules.reservation((float("inf"),) + tuple(th))))
    if sp.k == 2:
        res = [(ev((x,)), (x,)) for x in cand[0]]
        m, th = min(res, key=lambda r: r[0]["cost"])
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


def tune_reservation_sets(model, rules, start_th, max_passes=4):
    """Richer rule: per-class node subsets and thresholds, coordinate descent on the exact model,
    started from the tuned single-threshold rule."""
    sp = rules.sp
    vals = _lat_values(sp)
    N = sp.n
    subsets = [tuple(i for i in range(N) if (m >> i) & 1) for m in range(1, 2 ** N)]
    full = tuple(range(N))
    th = list(start_th)
    al = [full] * sp.k
    ev = lambda th, al: model.evaluate(model.policy_from_state_rule(rules.reservation_sets(th, al)))
    best = ev(th, al)
    for _ in range(max_passes):
        changed = False
        for c in range(sp.k):
            ths = [float("inf")] if c == 0 else [-1.0] + [v for v in vals if v <= sp.limits_ms[c]]
            for sub in subsets:
                for x in ths:
                    if sub == al[c] and x == th[c]:
                        continue
                    t2, a2 = list(th), list(al)
                    t2[c], a2[c] = x, sub
                    m = ev(t2, a2)
                    if m["cost"] < best["cost"] - 1e-12:
                        best, th, al, changed = m, t2, a2, True
        if not changed:
            break
    return th, al, best


def reference(sysp, sets=False):
    model = GeneralModel(sysp)
    rules = Rules(sysp)
    out = {"load": sysp.load(), "states": model.S * model.K * model.T}
    out["jsq"] = model.evaluate(model.policy_from_state_rule(rules.jsq))
    out["sla_rule"] = model.evaluate(model.policy_from_state_rule(rules.sla))
    th, m = tune_reservation(model, rules)
    out["reservation"] = {**m, "thresholds": [None if not np.isfinite(x) else x for x in th]}
    out["_reservation_th"] = th
    if sets:
        th2, al2, m2 = tune_reservation_sets(model, rules, th)
        out["reservation_sets"] = {**m2, "thresholds": [None if not np.isfinite(x) else x for x in th2],
                                   "allowed": [list(a) for a in al2]}
    pol, _ = model.solve(model.cost_eval())
    out["optimum"] = model.evaluate(pol)
    return model, out
