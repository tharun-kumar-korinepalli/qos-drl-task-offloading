import numpy as np
import pytest

from qos_offloading import OffloadingEnv, Reward, System
from qos_offloading import study as S
from qos_offloading.training import Job


def test_lagrange_update_moves_multiplier_towards_target():
    sp = System()
    env = OffloadingEnv(sp, Reward(kind="lagrangian", penalties=(10.0, 2.0), target=(0.02,), eta=10.0))
    env.reset(seed=0)
    env.block_n[:] = [100, 100]
    env.block_m[:] = [12, 30]                       # critical miss rate 0.12 > target 0.02
    v_hat, lam = env.lagrange_update()
    assert v_hat[0] == pytest.approx(0.12)
    assert lam[0] == pytest.approx(10.0 + 10.0 * (0.12 - 0.02))
    assert lam[1] == pytest.approx(2.0)              # least critical class is not a constraint
    assert env.block_n.sum() == 0 and env.block_m.sum() == 0
    env.block_n[:] = [100, 100]
    env.block_m[:] = [0, 0]
    _, lam = env.lagrange_update()
    assert lam[0] == pytest.approx(11.0 - 10.0 * 0.02)
    env.lam[0] = 0.05
    env.block_n[:] = [10, 0]
    _, lam = env.lagrange_update()
    assert lam[0] == 0.0                             # projected onto lambda >= 0


def test_lagrangian_reward_uses_current_multiplier():
    env = OffloadingEnv(System(), Reward(kind="lagrangian", penalties=(10.0, 2.0), target=(0.02,), eta=1.0))
    env.reset(seed=0)
    env.lam[:] = [7.0, 2.0]
    assert env._reward(0, 0, True) == pytest.approx(-7.0)
    assert env._reward(0, 1, True) == pytest.approx(-2.0)
    assert env._reward(env.N, 0, False) == pytest.approx(1.0 - 0.1)
    assert list(env.block_n) == [2, 1] and list(env.block_m) == [1, 1]


def test_targets_are_optimum_miss_rates():
    t = S.optimum_miss("nominal")
    assert t[0] == pytest.approx(0.020625, abs=1e-6)
    rw = S.lagrangian_reward("large4_3c", 1.0)
    assert len(rw.target) == 2 and rw.kind == "lagrangian"


def test_stored_runs_keep_their_cache_keys():
    # The revision code must not change the key of any stored run, or everything would be retrained.
    from qos_offloading.training import load
    best = S.select_configs(load)
    if best is None:
        pytest.skip("tuning runs not present")
    best_sys = S.select_system_configs(load)
    missing = [j.name() for j in list(S.main_jobs(best).values()) + list(S.system_jobs(best_sys).values())
               if not j.path().exists()]
    assert missing == []


def test_short_lagrangian_training_runs(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from qos_offloading import training
    monkeypatch.setattr(training, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(training, "TABLES", tmp_path / "tables")
    monkeypatch.setattr(training, "MODELS", tmp_path / "models")
    job = Job("ppo", System(), S.lagrangian_reward("nominal", 1.0), 0, 2048,
              algo_kwargs=(("n_steps", 512), ("batch_size", 64)), curve_every=1024)
    rec = __import__("json").loads(training.execute(job).read_text())
    assert len(rec["duals"]) == 4                   # one dual step per 512-step rollout
    assert len(rec["curve"]) == 2 and "multipliers" in rec["curve"][0]
    assert np.isfinite(rec["exact"]["cost"]) and len(rec["final_penalties"]) == 2
