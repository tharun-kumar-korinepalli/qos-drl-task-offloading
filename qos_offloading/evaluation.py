import numpy as np

from . import config as C
from .envs import EdgeCloudEnv


def evaluate_policy(policy_fn, env_cls=EdgeCloudEnv,
                    n_episodes=C.EVAL_EPISODES, max_steps=C.MAX_STEPS,
                    seed0=C.EVAL_SEED, critical_prob=0.5):
    violations = {c: 0 for c in C.TASK_CLASSES}
    counts = {c: 0 for c in C.TASK_CLASSES}
    latencies = {c: [] for c in C.TASK_CLASSES}
    edge_actions = 0
    total_actions = 0

    for episode in range(n_episodes):
        env = env_cls(max_steps=max_steps)
        obs, _ = env.reset(seed=seed0 + episode)
        done = False
        while not done:
            action = policy_fn(obs)
            obs, _reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            cls = info["task_class"]
            counts[cls] += 1
            latencies[cls].append(info["latency"])
            if info["violated"]:
                violations[cls] += 1
            if action in (C.EDGE_1, C.EDGE_2):
                edge_actions += 1
            total_actions += 1

    results = {}
    for cls in C.TASK_CLASSES:
        results[f"class_{cls}_violation_rate"] = violations[cls] / counts[cls]
        results[f"class_{cls}_avg_latency"] = float(np.mean(latencies[cls]))
    results["overall_violation_rate"] = sum(violations.values()) / sum(counts.values())
    results["sla_cost_per_task"] = (
        critical_prob * C.EVAL_PENALTY[C.CRITICAL] * results["class_0_violation_rate"]
        + (1.0 - critical_prob) * C.EVAL_PENALTY[C.ANALYTICS] * results["class_1_violation_rate"]
    )
    results["edge_utilization"] = edge_actions / total_actions
    return results


def summarize(results):
    return {
        m: (float(np.mean([r[m] for r in results])), float(np.std([r[m] for r in results])))
        for m in results[0]
    }


def print_results(title, results):
    print(f"=== {title} ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")


def print_summary(title, summary):
    print(f"=== {title} ===")
    for m, (mean, std) in summary.items():
        print(f"{m}: {mean:.4f} +/- {std:.4f}")
