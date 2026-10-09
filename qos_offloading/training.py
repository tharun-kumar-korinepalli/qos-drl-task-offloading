from stable_baselines3 import PPO

from . import config as C
from .evaluation import evaluate_policy


def train_ppo(env_cls, seed=None, total_timesteps=C.TOTAL_TIMESTEPS, verbose=0):
    env = env_cls()
    env.reset(seed=seed)
    model = PPO("MlpPolicy", env, verbose=verbose, seed=seed)
    model.learn(total_timesteps=total_timesteps)
    return model


def make_policy(model):
    return lambda obs: model.predict(obs, deterministic=True)[0]


def run_trial(train_env_cls, eval_env_cls, seed, total_timesteps=C.TOTAL_TIMESTEPS):
    model = train_ppo(train_env_cls, seed=seed, total_timesteps=total_timesteps)
    return evaluate_policy(make_policy(model), env_cls=eval_env_cls)


def run_multi_seed(train_env_cls, eval_env_cls, seeds=C.SEEDS,
                   total_timesteps=C.TOTAL_TIMESTEPS, label=None):
    results = []
    for s in seeds:
        prefix = f"[{label}] " if label else ""
        print(f"{prefix}seed {s}...")
        r = run_trial(train_env_cls, eval_env_cls, s, total_timesteps)
        print(" ", r)
        results.append(r)
    return results


def train_dqn(env_cls, seed=None, total_timesteps=C.TOTAL_TIMESTEPS, verbose=0, **kwargs):
    from stable_baselines3 import DQN

    env = env_cls()
    env.reset(seed=seed)
    model = DQN("MlpPolicy", env, verbose=verbose, seed=seed, **kwargs)
    model.learn(total_timesteps=total_timesteps)
    return model


def run_trial_dqn(train_env_cls, eval_env_cls, seed, total_timesteps=C.TOTAL_TIMESTEPS, **kwargs):
    model = train_dqn(train_env_cls, seed=seed, total_timesteps=total_timesteps, **kwargs)
    return evaluate_policy(make_policy(model), env_cls=eval_env_cls)


def run_multi_seed_dqn(train_env_cls, eval_env_cls, seeds=C.SEEDS,
                       total_timesteps=C.TOTAL_TIMESTEPS, label=None, **kwargs):
    results = []
    for s in seeds:
        prefix = f"[{label}] " if label else ""
        print(f"{prefix}seed {s}...")
        r = run_trial_dqn(train_env_cls, eval_env_cls, s, total_timesteps, **kwargs)
        print(" ", r)
        results.append(r)
    return results
