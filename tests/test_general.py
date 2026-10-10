import numpy as np
import pytest

from qos_offloading.dp import ExactModel, optimal_sla_policy, strip
from qos_offloading.envs import OffloadingEnv
from qos_offloading.general import (GeneralEnv, GeneralModel, RewardSpec, Rules, SysParams,
                                    corrected_penalties, reference)
from qos_offloading.params import EnvParams
from qos_offloading.studies_v3 import SCENARIOS


def simulate_table(sysp, table, episodes, seed0):
    env = GeneralEnv(sysp, RewardSpec(penalties=tuple(sysp.eval_weights)))
    K = sysp.k
    viol, cnt = np.zeros(K), np.zeros(K)
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        done = False
        while not done:
            s = int(np.asarray(env.q) @ (sysp.cap + 1) ** np.arange(sysp.n - 1, -1, -1))
            a = int(table[env.t, s, env.c])
            obs, r, done, _, info = env.step(a)
            cnt[info["task_class"]] += 1
            viol[info["task_class"]] += info["violated"]
    return viol / cnt


def test_default_system_equals_original_simulator():
    a, b = GeneralEnv(), OffloadingEnv(EnvParams())
    a.reset(seed=3)
    b.reset(seed=3)
    rng = np.random.default_rng(0)
    for i in range(1000):
        act = int(rng.integers(3))
        ra, rb = a.step(act), b.step(act)
        assert np.array_equal(ra[0], rb[0]) and ra[1] == pytest.approx(rb[1]) and ra[2] == rb[2]
        if ra[2]:
            a.reset(seed=i)
            b.reset(seed=i)


def test_default_model_equals_original_model():
    m = GeneralModel()
    pol, _ = m.solve(m.cost_eval())
    old = ExactModel(EnvParams().eval_view())
    ref = strip(old.evaluate(optimal_sla_policy(old)))
    assert m.evaluate(pol)["cost"] == pytest.approx(ref["sla_cost_per_task"], abs=1e-12)


@pytest.mark.parametrize("name", ["hetero3", "hetero3_3c"])
def test_exact_model_matches_simulation(name):
    sp = SCENARIOS[name]
    m = GeneralModel(sp)
    rng = np.random.default_rng(1)
    table = rng.integers(0, sp.n + 1, size=(m.T, m.S, m.K)).astype(np.int8)
    ex = m.evaluate(table)["miss"]
    mc = simulate_table(sp, table, 400, 9000)
    assert np.allclose(ex, mc, atol=0.015)  # about three standard errors for a random table
    pol, _ = m.solve(m.cost_eval())
    ex = m.evaluate(pol)["miss"]
    mc = simulate_table(sp, pol, 400, 9500)
    assert np.allclose(ex, mc, atol=0.01)


def test_transition_rows_sum_to_one():
    m = GeneralModel(SCENARIOS["hetero3"])
    for mats in m.P.values():
        for M in mats:
            assert np.allclose(np.asarray(M.sum(axis=1)).ravel(), 1.0)


def test_corrected_penalties():
    w = corrected_penalties((10.0, 2.0), (0.5, 0.5), 6.0)
    assert (w[0] + 1) / (w[1] + 1) == pytest.approx(5.0)
    assert 0.5 * w[0] + 0.5 * w[1] == pytest.approx(6.0)


def test_corrected_reward_policy_is_optimal_on_nominal_system():
    # with the success bonus folded in, the reward has the operator's ratio; only the cloud cost differs
    m = GeneralModel()
    opt = m.evaluate(m.solve(m.cost_eval())[0])["cost"]
    r = RewardSpec(penalties=corrected_penalties((10.0, 2.0), (0.5, 0.5), 6.0))
    got = m.evaluate(m.solve(m.cost_reward(r))[0])["cost"]
    assert got == pytest.approx(opt, rel=1e-6)


def test_optimum_lower_bounds_rules():
    sp = SCENARIOS["hetero3"]
    model, ref = reference(sp, sets=False)
    for k in ["jsq", "sla_rule", "reservation"]:
        assert ref["optimum"]["cost"] <= ref[k]["cost"] + 1e-12


def test_expected_return_matches_simulation():
    sp = SCENARIOS["hetero3"]
    m = GeneralModel(sp)
    r = RewardSpec(penalties=(10.0, 2.0))
    pol, _ = m.solve(m.cost_reward(r))
    exact = m.expected_return(pol, r)
    env = GeneralEnv(sp, r)
    tot = []
    for ep in range(300):
        env.reset(seed=500 + ep)
        g, done = 0.0, False
        while not done:
            s = int(np.asarray(env.q) @ (sp.cap + 1) ** np.arange(sp.n - 1, -1, -1))
            _, rew, done, _, _ = env.step(int(pol[env.t, s, env.c]))
            g += rew
        tot.append(g)
    assert abs(np.mean(tot) - exact) < 3 * np.std(tot) / np.sqrt(len(tot))


def test_pruning_mask_and_features():
    env = GeneralEnv(SysParams(), RewardSpec(), svrs_features=True, prune=True)
    env.reset(seed=0)
    env.q[:] = [8, 0]
    env.c = 0
    m = env.action_masks()
    assert list(m) == [False, True, True]
    assert env._obs().shape == env.observation_space.shape
