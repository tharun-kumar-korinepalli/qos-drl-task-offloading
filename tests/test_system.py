import numpy as np
import pytest

from qos_offloading import ExactModel, OffloadingEnv, Reward, Rules, System, corrected_penalties, reference
from qos_offloading.rules import tune_reservation
from qos_offloading.study import SYSTEMS
from qos_offloading.training import Job


def state_index(sp, q):
    return int(np.asarray(q) @ (sp.cap + 1) ** np.arange(sp.n - 1, -1, -1))


def simulate_table(sp, table, episodes, seed0):
    env = OffloadingEnv(sp, Reward(penalties=tuple(sp.eval_weights)))
    viol, cnt = np.zeros(sp.k), np.zeros(sp.k)
    for ep in range(episodes):
        env.reset(seed=seed0 + ep)
        done = False
        while not done:
            _, _, done, _, info = env.step(int(table[env.t, state_index(sp, env.q), env.c]))
            cnt[info["task_class"]] += 1
            viol[info["task_class"]] += info["violated"]
    return viol / cnt


def test_nominal_optimum():
    m = ExactModel()
    assert m.evaluate(m.solve(m.cost_eval())[0])["cost"] == pytest.approx(0.26137243777058977, abs=1e-12)


@pytest.mark.parametrize("sp", [System(), SYSTEMS["hetero3"], SYSTEMS["hetero3_3c"]])
def test_exact_model_matches_simulation(sp):
    m = ExactModel(sp)
    rng = np.random.default_rng(1)
    table = rng.integers(0, sp.n + 1, size=(m.T, m.S, m.K)).astype(np.int8)
    assert np.allclose(m.evaluate(table)["miss"], simulate_table(sp, table, 400, 9000), atol=0.015)
    pol, _ = m.solve(m.cost_eval())
    assert np.allclose(m.evaluate(pol)["miss"], simulate_table(sp, pol, 400, 9500), atol=0.01)


def test_transition_rows_sum_to_one():
    for sp in [System(), SYSTEMS["hetero3"]]:
        for mats in ExactModel(sp).P.values():
            for M in mats:
                assert np.allclose(np.asarray(M.sum(axis=1)).ravel(), 1.0)


def test_background_schedule_matches_simulator():
    sp = System(drain=(0.0, 0.0))
    env = OffloadingEnv(sp)
    env.reset(seed=0)
    seen = []
    for _ in range(40):
        before = env.q.sum()
        env.step(sp.n)
        seen.append(int(env.q.sum() - before))
    assert sum(seen) == 7


def test_corrected_penalties():
    w = corrected_penalties((10.0, 2.0), (0.5, 0.5), 6.0)
    assert (w[0] + 1) / (w[1] + 1) == pytest.approx(5.0)
    assert 0.5 * w[0] + 0.5 * w[1] == pytest.approx(6.0)


def test_corrected_reward_is_optimal_on_nominal_system():
    m = ExactModel()
    opt = m.evaluate(m.solve(m.cost_eval())[0])["cost"]
    r = Reward(penalties=corrected_penalties((10.0, 2.0), (0.5, 0.5), 6.0))
    assert m.evaluate(m.solve(m.cost_reward(r))[0])["cost"] == pytest.approx(opt, rel=1e-6)


def test_optimum_lower_bounds_rules():
    _, ref, _ = reference(SYSTEMS["hetero3"])
    for k in ["jsq", "sla_rule", "reservation"]:
        assert ref["optimum"]["cost"] <= ref[k]["cost"] + 1e-12


def test_nominal_reservation_threshold():
    th, m = tune_reservation(ExactModel(), Rules(System()))
    assert th[1] == 30.0 and m["cost"] == pytest.approx(0.3042, abs=1e-4)


def test_expected_return_matches_simulation():
    sp = SYSTEMS["hetero3"]
    m = ExactModel(sp)
    r = Reward(penalties=(10.0, 2.0))
    pol, _ = m.solve(m.cost_reward(r))
    env = OffloadingEnv(sp, r)
    totals = []
    for ep in range(300):
        env.reset(seed=500 + ep)
        g, done = 0.0, False
        while not done:
            _, rew, done, _, _ = env.step(int(pol[env.t, state_index(sp, env.q), env.c]))
            g += rew
        totals.append(g)
    assert abs(np.mean(totals) - m.expected_return(pol, r)) < 3 * np.std(totals) / np.sqrt(len(totals))


def test_observations_match_env():
    m = ExactModel()
    env = OffloadingEnv()
    obs, _ = env.reset(seed=0)
    assert m.observations()[int(obs[2])].tolist() == obs.tolist()


def test_pruning_mask_and_risk_features():
    env = OffloadingEnv(System(), Reward(), risk_features=True, prune=True)
    env.reset(seed=0)
    env.q[:] = [8, 0]
    env.c = 0
    assert list(env.action_masks()) == [False, True, True]
    assert env._obs().shape == env.observation_space.shape


def test_job_key_depends_on_spec_only():
    a = Job("ppo", System(), Reward(), 0, 1000, tag="a")
    b = Job("ppo", System(), Reward(), 0, 1000, tag="b")
    assert a.key() == b.key()
    assert Job("ppo", System(), Reward(kind="flat"), 0, 1000).key() != a.key()
