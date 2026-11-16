import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from qos_offloading import ExactModel
from qos_offloading import study as S

OUT = ROOT / "results" / "summary" / "nominal_dp_under_shifts.json"


def main():
    nominal = ExactModel(S.NOMINAL)
    policy, _ = nominal.solve(nominal.cost_eval())
    rows = {}
    for name, system in S.SHIFTS.items():
        model = ExactModel(system)
        opt = model.evaluate(model.solve(model.cost_eval())[0])
        ev = model.evaluate(policy)
        rows[name] = {"optimum": opt["cost"], "nominal_dp": ev["cost"],
                      "gap_pct": 100 * (ev["cost"] / opt["cost"] - 1), "nominal_dp_miss": list(ev["miss"])}
        print(name, rows[name], flush=True)
    OUT.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
