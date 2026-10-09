import json

import numpy as np
import pytest

from qos_offloading import stats
from qos_offloading.params import FLAT, WEIGHTED, EnvParams
from qos_offloading.runner import Job


def test_holm_matches_definition():
    adj = stats.holm([0.01, 0.04, 0.03])
    assert adj.tolist() == pytest.approx([0.03, 0.06, 0.06])


def test_a12_extremes():
    assert stats.vargha_delaney_a12([3, 4], [1, 2]) == 1.0
    assert stats.vargha_delaney_a12([1, 2], [3, 4]) == 0.0
    assert stats.vargha_delaney_a12([1, 2], [1, 2]) == 0.5


def test_mann_whitney_fully_separated_five_vs_five():
    assert stats.mann_whitney([1, 2, 3, 4, 5], [6, 7, 8, 9, 10]) == pytest.approx(2 / 252)


def test_bootstrap_ci_contains_mean():
    x = np.random.default_rng(0).normal(size=30)
    lo, hi = stats.bootstrap_ci(x)
    assert lo < x.mean() < hi


def test_sign_test():
    better, worse, p = stats.sign_test_vs_constant([1, 1, 1, 1, 1, 1], 2)
    assert (better, worse) == (6, 0) and p == pytest.approx(2 / 64)


def test_job_key_is_stable_and_canonical():
    a = Job("ppo", FLAT, 0, 20_000)
    b = Job("ppo", EnvParams(critical_penalty=6.0, analytics_penalty=6.0), 0, 20_000)
    assert a.key() == b.key()
    assert Job("ppo", WEIGHTED, 0, 20_000).key() != a.key()
    assert json.dumps(a.spec(), sort_keys=True)


def test_eval_view_uses_evaluation_weights():
    v = FLAT.eval_view()
    assert (v.critical_penalty, v.analytics_penalty, v.flat) == (10.0, 2.0, False)
