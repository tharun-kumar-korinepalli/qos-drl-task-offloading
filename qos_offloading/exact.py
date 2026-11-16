from itertools import product

import numpy as np
from scipy import sparse

from .system import System, background_counts, background_dist


class ExactModel:
    def __init__(self, system=System()):
        self.sp = system
        self.N, self.K, self.T = system.n, system.k, system.max_steps
        self.Q = system.cap + 1
        self.S = self.Q ** self.N
        self.A = self.N + 1
        self.p = np.array(system.class_probs, dtype=float)
        self.qs = np.indices([self.Q] * self.N).reshape(self.N, -1).T
        self.radix = self.Q ** np.arange(self.N - 1, -1, -1)
        lat = np.zeros((self.S, self.A))
        for i in range(self.N):
            lat[:, i] = system.base_ms[i] + system.slope_ms[i] * self.qs[:, i]
        lat[:, self.N] = system.cloud_ms
        self.latency = lat
        miss = (lat[:, :, None] > np.array(system.limits_ms)[None, None, :]).astype(float)
        miss[:, self.N, :] = 1.0 - np.array(system.cloud_meet)[None, :]
        self.miss = miss
        if system.background_mode == "deterministic":
            self.sched = background_counts(system)
            self.P = {k: self._build({k: 1.0}) for k in sorted(set(self.sched.tolist()))}
        else:
            self.P = {"r": self._build(background_dist(system))}
        self.PT = {key: sparse.hstack([m.T for m in mats]).tocsr() for key, mats in self.P.items()}

    def _build(self, kdist):
        S, N = self.S, self.N
        d = np.array(self.sp.drain)
        patterns = np.array(list(product([0, 1], repeat=N)), dtype=int)
        pprob = np.prod(np.where(patterns == 1, d[None, :], 1.0 - d[None, :]), axis=1)
        mats = []
        for a in range(self.A):
            rows, cols, vals = [], [], []
            for kb, pk in kdist.items():
                q = self.qs.copy()
                if a < N:
                    q[:, a] = np.minimum(q[:, a] + 1, self.sp.cap)
                for _ in range(kb):
                    j = np.argmin(q, axis=1)
                    q[np.arange(S), j] = np.minimum(q[np.arange(S), j] + 1, self.sp.cap)
                for pat, pp in zip(patterns, pprob):
                    rows.append(np.arange(S))
                    cols.append(np.maximum(q - pat[None, :], 0) @ self.radix)
                    vals.append(np.full(S, pk * pp))
            M = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(S, S))
            M.sum_duplicates()
            mats.append(M)
        return mats

    def _key(self, t):
        return int(self.sched[t]) if self.sp.background_mode == "deterministic" else "r"

    def cost_eval(self, weights=None):
        w = np.array(weights if weights is not None else self.sp.eval_weights, dtype=float)
        return self.miss * w[None, None, :]

    def cost_reward(self, reward):
        w = np.array(reward.effective_penalties(self.sp), dtype=float)
        c = self.miss * w[None, None, :] - reward.success * (1.0 - self.miss)
        c[:, self.N, :] += reward.cloud_cost
        return c

    def solve(self, cost, gamma=1.0):
        pol = np.zeros((self.T, self.S, self.K), dtype=np.int8)
        V = np.zeros((self.S, self.K))
        for t in range(self.T - 1, -1, -1):
            W = V @ self.p
            cont = np.stack([m @ W for m in self.P[self._key(t)]], axis=1)
            Qv = cost + gamma * cont[:, :, None]
            pol[t] = np.argmin(Qv, axis=1)
            V = np.min(Qv, axis=1)
        return pol, float(V[0] @ self.p)

    def _forward(self, policy, per_step):
        S, A = self.S, self.A
        dist = np.zeros((S, self.K))
        dist[0] = self.p
        rows = np.arange(S)
        for t in range(self.T):
            act = policy[t].astype(np.int64)
            per_step(dist, act)
            v = np.zeros(A * S)
            for c in range(self.K):
                np.add.at(v, act[:, c] * S + rows, dist[:, c])
            dist = (self.PT[self._key(t)] @ v)[:, None] * self.p[None, :]

    def evaluate(self, policy):
        K = self.K
        rows = np.arange(self.S)
        acc = {"viol": np.zeros(K), "cnt": np.zeros(K), "lat": np.zeros(K), "edge": 0.0}

        def step(dist, act):
            for c in range(K):
                a, w = act[:, c], dist[:, c]
                acc["cnt"][c] += w.sum()
                acc["viol"][c] += w @ self.miss[rows, a, c]
                acc["lat"][c] += w @ self.latency[rows, a]
                acc["edge"] += w @ (a != self.N)

        self._forward(policy, step)
        vr = acc["viol"] / acc["cnt"]
        return {"miss": vr.tolist(), "latency": (acc["lat"] / acc["cnt"]).tolist(),
                "overall": float(acc["viol"].sum() / acc["cnt"].sum()),
                "cost": float(self.p @ (np.array(self.sp.eval_weights) * vr)),
                "edge": float(acc["edge"] / acc["cnt"].sum())}

    def expected_return(self, policy, reward):
        c = self.cost_reward(reward)
        rows = np.arange(self.S)
        total = [0.0]

        def step(dist, act):
            for k in range(self.K):
                total[0] -= dist[:, k] @ c[rows, act[:, k], k]

        self._forward(policy, step)
        return float(total[0])

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

    def policy_from_rule(self, rule):
        acts = np.zeros((self.S, self.K), dtype=np.int8)
        for s in range(self.S):
            for c in range(self.K):
                acts[s, c] = rule(self.qs[s], c)
        return np.broadcast_to(acts, (self.T, self.S, self.K)).copy()

    def frontier(self, n=101):
        pts = []
        for lam in np.linspace(0.0, 1.0, n):
            pol, _ = self.solve(self.cost_eval((lam / self.p[0], (1 - lam) / self.p[1])))
            m = self.evaluate(pol)
            pts.append((float(lam), m["miss"][0], m["miss"][1]))
        return pts
