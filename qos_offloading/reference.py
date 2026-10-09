from . import config as C
from .baselines import ReservationBaseline, ShortestQueueBaseline, SLAThresholdBaseline
from .dp import ExactModel, optimal_reward_policy, optimal_sla_policy, pareto_frontier, strip


def sla_of(params):
    return {C.CRITICAL: params.critical_sla, C.ANALYTICS: params.analytics_sla}


def tune_reservation(model, params):
    best = None
    for k in range(C.SCARCE_QUEUE_CAP + 1):
        m = strip(model.evaluate(model.policy_from_obs_rule(ReservationBaseline(sla_of(params), k))))
        if best is None or m["sla_cost_per_task"] < best[1]["sla_cost_per_task"] - 1e-12:
            best = (k, m)
    return best


def reference_policies(params, frontier_points=0):
    ev = params.eval_view()
    model = ExactModel(ev)
    out = {}
    out["rr"] = strip(model.evaluate_mixture(model.round_robin_policies()))
    out["jsq"] = strip(model.evaluate(model.policy_from_obs_rule(ShortestQueueBaseline())))
    out["threshold"] = strip(model.evaluate(model.policy_from_obs_rule(SLAThresholdBaseline(sla_of(ev)))))
    k, m = tune_reservation(model, ev)
    out["reservation"] = {**m, "analytics_queue_limit": k}
    out["dp_sla_optimal"] = strip(model.evaluate(optimal_sla_policy(model)))
    out["dp_reward_weighted"] = strip(model.evaluate(optimal_reward_policy(model, ev)))
    flat = ev.__class__(**{**ev.to_dict(), "flat": True})
    out["dp_reward_flat"] = strip(model.evaluate(optimal_reward_policy(model, flat)))
    if frontier_points:
        out["frontier"] = pareto_frontier(model, frontier_points)
    return out
