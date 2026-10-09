import numpy as np
import pytest

from qos_offloading import config as C
from qos_offloading.baselines import (ReservationBaseline, ShortestQueueBaseline, SLAThresholdBaseline,
                                      TablePolicy)
from qos_offloading.dp import (ExactModel, background_schedule, optimal_reward_policy, optimal_sla_policy,
                               pareto_frontier, strip)
from qos_offloading.envs import EdgeCloudEnvScarce, EdgeCloudEnvScarceClassBlind, OffloadingEnv
from qos_offloading.evaluation import evaluate_policy
from qos_offloading.params import FLAT, WEIGHTED, EnvParams

SLA = {C.CRITICAL: C.CRITICAL_SLA_MS, C.ANALYTICS: C.ANALYTICS_SLA_SCARCE_MS}


@pytest.fixture(scope="module")
def model():
    return ExactModel(WEIGHTED)


def test_parametric_env_matches_legacy_envs():
    for params, legacy in [(WEIGHTED, EdgeCloudEnvScarce), (FLAT, EdgeCloudEnvScarceClassBlind)]:
        a, b = OffloadingEnv(params), legacy()
        a.reset(seed=3)
        b.reset(seed=3)
        rng = np.random.default_rng(0)
        for _ in range(600):
            act = int(rng.integers(3))
            ra, rb = a.step(act), b.step(act)
            assert np.array_equal(ra[0], rb[0]) and ra[1] == rb[1] and ra[4] == rb[4]
            if ra[2]:
                a.reset(seed=1)
                b.reset(seed=1)


def test_flat_penalty_matches_class_mix():
    for p in (0.3, 0.5, 0.7):
        f = EnvParams(critical_prob=p, flat=True)
        assert f.penalties()[0] == pytest.approx(p * 10 + (1 - p) * 2)
    assert FLAT.penalties() == (6.0, 6.0)


def test_canonical_flat_equals_equal_penalties():
    assert FLAT.canonical() == EnvParams(critical_penalty=6.0, analytics_penalty=6.0)


def test_background_schedule_matches_simulator():
    env = OffloadingEnv(WEIGHTED)
    env.reset(seed=0)
    env.drain_prob = 0.0
    seen = []
    for _ in range(40):
        before = sum(env.edge_queues)
        env.step(C.CLOUD)
        seen.append(sum(env.edge_queues) - before)
    assert seen == background_schedule(WEIGHTED)[:40].tolist()
    assert sum(seen) == 7


def test_transition_rows_are_distributions(model):
    for P in model.P.values():
        assert np.allclose(P.sum(axis=2), 1.0)
        assert (P >= 0).all()


@pytest.mark.parametrize("rule", [ShortestQueueBaseline(), SLAThresholdBaseline(SLA), ReservationBaseline(SLA, 1)])
def test_exact_evaluation_matches_monte_carlo(model, rule):
    ex = strip(model.evaluate(model.policy_from_obs_rule(rule)))
    mc = evaluate_policy(rule, env_cls=EdgeCloudEnvScarce, n_episodes=1500, seed0=50_000)
    for k in ("class_0_violation_rate", "class_1_violation_rate", "edge_utilization"):
        assert abs(ex[k] - mc[k]) < 0.01


def test_round_robin_exact_matches_monte_carlo(model):
    from qos_offloading.baselines import RoundRobinBaseline
    ex = strip(model.evaluate_mixture(model.round_robin_policies()))
    mc = evaluate_policy(RoundRobinBaseline(), env_cls=EdgeCloudEnvScarce, n_episodes=1500, seed0=50_000)
    assert abs(ex["class_0_violation_rate"] - mc["class_0_violation_rate"]) < 0.01
    assert ex["edge_utilization"] == pytest.approx(2 / 3, abs=1e-3)


def test_dp_policy_simulates_to_its_exact_value(model):
    table = optimal_sla_policy(model)
    ex = strip(model.evaluate(table))
    mc = evaluate_policy(TablePolicy(table), env_cls=EdgeCloudEnvScarce, n_episodes=1500, seed0=70_000)
    assert abs(ex["sla_cost_per_task"] - mc["sla_cost_per_task"]) < 0.02


def test_dp_optimum_is_a_lower_bound(model):
    opt = strip(model.evaluate(optimal_sla_policy(model)))["sla_cost_per_task"]
    for rule in [ShortestQueueBaseline(), SLAThresholdBaseline(SLA)] + [ReservationBaseline(SLA, k) for k in range(9)]:
        c = strip(model.evaluate(model.policy_from_obs_rule(rule)))["sla_cost_per_task"]
        assert opt <= c + 1e-9
    for p in (WEIGHTED, FLAT):
        c = strip(model.evaluate(optimal_reward_policy(model, p)))["sla_cost_per_task"]
        assert opt <= c + 1e-9


def test_weighted_reward_optimum_beats_flat_reward_optimum(model):
    w = strip(model.evaluate(optimal_reward_policy(model, WEIGHTED)))
    f = strip(model.evaluate(optimal_reward_policy(model, FLAT)))
    assert w["class_0_violation_rate"] < f["class_0_violation_rate"]
    assert w["sla_cost_per_task"] < f["sla_cost_per_task"]


def test_frontier_is_monotone(model):
    pts = pareto_frontier(model, 21)
    crit = [c for _, c, _ in pts[1:]]
    ana = [a for _, _, a in pts[1:]]
    assert all(x >= y - 1e-9 for x, y in zip(crit, crit[1:]))
    assert all(x <= y + 1e-9 for x, y in zip(ana, ana[1:]))


def test_observation_grid_matches_env_encoding(model):
    obs = model.all_observations()
    env = OffloadingEnv(WEIGHTED)
    env.reset(seed=0)
    o0 = env._get_obs()
    assert obs.dtype == np.float32
    assert obs[(0 * model.S + 0) * 2 + int(o0[2])].tolist() == o0.tolist()
