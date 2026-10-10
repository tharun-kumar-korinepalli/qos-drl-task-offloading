import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from qos_offloading import ExactModel, Reward, corrected_penalties

OUT = ROOT / "results" / "summary" / "reward_analysis.json"


def operator_cost(m, r):
    return 2.0 * (0.5 * r * m["miss"][0] + 0.5 * m["miss"][1])


def main():
    model = ExactModel()
    dp = lambda rw, gamma=1.0: model.evaluate(model.solve(model.cost_reward(rw), gamma=gamma)[0])
    out = {"effective": {
        "weighted 10:2": dp(Reward(penalties=(10.0, 2.0))),
        "costs 11:3 with cloud cost, no bonus": dp(Reward(penalties=(11.0, 3.0), success=0.0)),
        "weighted 10:2, no cloud cost": dp(Reward(penalties=(10.0, 2.0), cloud_cost=0.0)),
        "weighted 9:1": dp(Reward(penalties=(9.0, 1.0))),
        "flat 6:6": dp(Reward(penalties=(6.0, 6.0))),
        "optimum": model.evaluate(model.solve(model.cost_eval())[0])}}
    out["discount"] = {str(g): {"weighted": dp(Reward(penalties=(10.0, 2.0)), g),
                                "flat": dp(Reward(penalties=(6.0, 6.0)), g)} for g in (1.0, 0.99, 0.95, 0.9)}
    flat = dp(Reward(penalties=(6.0, 6.0)))
    ratios = {}
    for r in (1, 2, 3, 5, 10, 20, 50):
        wa = 12.0 / (r + 1)
        corr = corrected_penalties((float(r), 1.0), (0.5, 0.5), 6.0)
        opt = model.evaluate(model.solve(model.cost_eval((float(r), 1.0)))[0])
        matched = dp(Reward(penalties=(r * wa, wa)))
        corrected = dp(Reward(penalties=corr))
        ratios[str(r)] = {"wc": r * wa, "wa": wa, "wc_corr": corr[0], "wa_corr": corr[1],
                          "J_opt": operator_cost(opt, r), "J_matched": operator_cost(matched, r),
                          "J_corrected": operator_cost(corrected, r), "J_flat": operator_cost(flat, r)}
    out["ratios"] = ratios
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
