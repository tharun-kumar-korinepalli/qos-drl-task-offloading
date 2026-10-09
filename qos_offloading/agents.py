import time

from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback

from . import config as C
from .dp import ExactModel, strip
from .envs import OffloadingEnv

ALGOS = {
    "ppo": (PPO, {}),
    "dqn": (DQN, dict(C.DQN_TUNED_KWARGS)),
    "a2c": (A2C, {}),
}


def build_model(algo, params, seed, algo_kwargs=None):
    cls, defaults = ALGOS[algo]
    env = OffloadingEnv(params)
    env.reset(seed=seed)
    kwargs = {**defaults, **(algo_kwargs or {})}
    return cls("MlpPolicy", env, verbose=0, seed=seed, **kwargs)


def policy_fn(model):
    return lambda obs: model.predict(obs, deterministic=True)[0]


def exact_metrics(model, exact):
    return strip(exact.evaluate(exact.policy_from_fn(policy_fn(model))))


class ExactEvalCallback(BaseCallback):
    def __init__(self, exact, every):
        super().__init__()
        self.exact = exact
        self.every = every
        self.curve = []

    def _on_step(self):
        if self.num_timesteps % self.every == 0:
            m = exact_metrics(self.model, self.exact)
            self.curve.append({"timesteps": int(self.num_timesteps), **m})
        return True


def train_and_evaluate(algo, train_params, seed, timesteps, eval_params=None,
                       algo_kwargs=None, curve_every=None, mc_episodes=C.MC_EVAL_EPISODES):
    from .evaluation import evaluate_policy

    eval_params = (eval_params or train_params).eval_view()
    exact = ExactModel(eval_params)
    model = build_model(algo, train_params, seed, algo_kwargs)
    cb = ExactEvalCallback(exact, curve_every) if curve_every else None
    t0 = time.time()
    model.learn(total_timesteps=timesteps, callback=cb)
    train_time = time.time() - t0
    ex = exact_metrics(model, exact)
    mc = evaluate_policy(policy_fn(model), env_cls=lambda max_steps: OffloadingEnv(eval_params),
                         n_episodes=mc_episodes, seed0=C.MC_EVAL_SEED, critical_prob=eval_params.critical_prob)
    return {"exact": ex, "mc": mc, "curve": cb.curve if cb else None,
            "train_seconds": train_time, "timesteps_done": int(model.num_timesteps)}

