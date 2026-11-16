import numpy as np
import pytest

from qos_offloading import stats


def test_holm():
    assert stats.holm([0.01, 0.04, 0.03]).tolist() == pytest.approx([0.03, 0.06, 0.06])


def test_a12():
    assert stats.vargha_delaney_a12([3, 4], [1, 2]) == 1.0
    assert stats.vargha_delaney_a12([1, 2], [3, 4]) == 0.0
    assert stats.vargha_delaney_a12([1, 2], [1, 2]) == 0.5


def test_mann_whitney_separated():
    assert stats.mann_whitney([1, 2, 3, 4, 5], [6, 7, 8, 9, 10]) == pytest.approx(2 / 252)


def test_bootstrap_ci_contains_mean():
    x = np.random.default_rng(0).normal(size=30)
    lo, hi = stats.bootstrap_ci(x)
    assert lo < x.mean() < hi


def test_sign_test():
    better, worse, p = stats.sign_test_vs_constant([1] * 6, 2)
    assert (better, worse) == (6, 0) and p == pytest.approx(2 / 64)
