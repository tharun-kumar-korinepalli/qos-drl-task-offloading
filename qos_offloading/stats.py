import numpy as np
from scipy import stats as st


def bootstrap_ci(x, n_boot=10_000, alpha=0.05, seed=0, stat=np.mean):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        v = float(stat(x))
        return v, v
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    b = stat(x[idx], axis=1)
    return float(np.quantile(b, alpha / 2)), float(np.quantile(b, 1 - alpha / 2))


def iqm(x, axis=None):
    x = np.sort(np.asarray(x, dtype=float), axis=-1 if axis is None else axis)
    n = x.shape[-1]
    lo, hi = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    return np.mean(x[..., lo:hi], axis=-1)


def vargha_delaney_a12(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    gt = (a[:, None] > b[None, :]).sum()
    eq = (a[:, None] == b[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(a) * len(b)))


def mann_whitney(a, b):
    method = "exact" if len(a) + len(b) <= 40 and len(set(a) | set(b)) == len(a) + len(b) else "asymptotic"
    return float(st.mannwhitneyu(a, b, alternative="two-sided", method=method).pvalue)


def sign_test_vs_constant(a, c):
    a = np.asarray(a, dtype=float)
    better = int((a < c).sum())
    worse = int((a > c).sum())
    n = better + worse
    p = float(st.binomtest(better, n, 0.5).pvalue) if n else 1.0
    return better, worse, p


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def describe(x):
    x = np.asarray(x, dtype=float)
    lo, hi = bootstrap_ci(x)
    return {"n": int(len(x)), "mean": float(x.mean()), "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            "ci_low": lo, "ci_high": hi, "median": float(np.median(x)), "iqm": float(iqm(x)),
            "min": float(x.min()), "max": float(x.max())}
