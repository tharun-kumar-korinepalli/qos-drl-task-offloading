from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def _dict(obj):
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(obj).items()}


@dataclass(frozen=True)
class System:
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
        return _dict(self)


@dataclass(frozen=True)
class Reward:
    kind: str = "weighted"
    penalties: tuple = (10.0, 2.0)
    success: float = 1.0
    cloud_cost: float = 0.1
    scale: float = 1.0
    target: tuple = ()
    eta: float = 0.0
    window: int = 200

    def effective_penalties(self, system):
        if self.kind == "flat":
            m = float(np.dot(system.class_probs, self.penalties))
            return tuple([m] * system.k)
        return tuple(float(x) for x in self.penalties)

    def to_dict(self):
        return _dict(self)


def corrected_penalties(weights, probs, mean_penalty, success=1.0):
    w = np.asarray(weights, dtype=float)
    p = np.asarray(probs, dtype=float)
    a = (mean_penalty + success) / float(p @ w)
    return tuple(float(x) for x in a * w - success)


def background_counts(system):
    pending = 0.0
    out = []
    for _ in range(system.max_steps):
        pending += (1.0 + system.background) - 1.0
        k = 0
        while pending >= 1.0:
            k += 1
            pending -= 1.0
        out.append(k)
    return np.array(out, dtype=int)


def background_dist(system):
    lo = int(np.floor(system.background))
    frac = system.background - lo
    if frac < 1e-12:
        return {lo: 1.0}
    return {lo: 1.0 - frac, lo + 1: frac}


class OffloadingEnv(gym.Env):
    def __init__(self, system=System(), reward=Reward(), risk_features=False, prune=False):
        super().__init__()
        self.sp = system
        self.rw = reward
        self.risk = risk_features
        self.prune = prune
        self.N, self.K = system.n, system.k
        self.base = np.array(system.base_ms)
        self.slope = np.array(system.slope_ms)
        self.limits = np.array(system.limits_ms)
        self.pen = np.array(reward.effective_penalties(system))
        self.lam = self.pen.copy()
        self.hist = [[] for _ in range(self.K)]
        hi = [system.cap] * self.N + [self.K - 1, 1]
        lo = [0] * (self.N + 2)
        if self.risk:
            hi += [4.0] * self.N + [1.0] * self.K
            lo += [0.0] * (self.N + self.K)
        self.observation_space = spaces.Box(low=np.array(lo, dtype=np.float32), high=np.array(hi, dtype=np.float32))
        self.action_space = spaces.Discrete(self.N + 1)
        self.sched = background_counts(system) if system.background_mode == "deterministic" else None
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.q = np.zeros(self.N, dtype=int)
        self.t = 0
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
        if self.risk:
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
        return self._obs(), r, done, False, {"latency": float(lat), "violated": miss, "task_class": c}

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

    def _remember(self, c, miss):
        self.hist[c].append(1.0 if miss else 0.0)
        if len(self.hist[c]) > self.rw.window:
            self.hist[c].pop(0)

    def _reward(self, a, c, miss):
        rw = self.rw
        if rw.kind == "tiered_adaptive":
            pen = self.lam[c]
            self._remember(c, miss)
            self.lam[c] = max(0.0, self.lam[c] + rw.eta * (float(np.mean(self.hist[c])) - rw.target[c]))
        else:
            pen = self.pen[c]
            if self.risk:
                self._remember(c, miss)
        r = -pen if miss else rw.success
        if a == self.N:
            r -= rw.cloud_cost
        return rw.scale * r
