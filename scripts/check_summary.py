"""Print the paper claims that report.py does not show. Run after analyze.py:
    python scripts/check_summary.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
S = json.loads((ROOT / "results" / "summary" / "summary.json").read_text())
REF = json.loads((ROOT / "results" / "summary" / "references.json").read_text())


def pct(x):
    return f"{100 * x:.1f}"


print("== Sec. 5.3 / Table 4: published design vs fixed-reward PPO (Holm over 6 tests)")
for k, v in S["published"]["rows"].items():
    for rw in ["weighted", "flat"]:
        t = v[f"vs_ppo_{rw}"]
        print(f"  {k:28s} vs {rw:8s} p_holm={t.get('p_holm', float('nan')):.2e}  lower={t['lower']}/{t['n']}  A12={t['a12']:.2f}")
    print(f"  {k:28s} beats reservation in {v['vs_reservation']['better']}/10 seeds")

print("\n== Table 5: per-class miss rates (%) and seeds beating the SLA rule")
for k, v in S["systems"]["rows"].items():
    miss = [pct(v[f"miss{c}"]["mean"]) for c in range(3) if f"miss{c}" in v]
    print(f"  {k:22s} miss={', '.join(miss):18s} beats SLA rule {v['vs_sla_rule']['better']}/10")

print("\n== Sec. 5.2: overall miss rate, edge share (paper: flat PPO 6.0%, weighted 9.0%, edge >= 89%)")
for k, v in S["nominal"]["rows"].items():
    print(f"  {k:15s} overall={pct(v['overall']['mean'])}%  min edge={pct(v['edge']['min'])}%  "
          f"lat crit/ana={v['lat0']['mean']:.1f}/{v['lat1']['mean']:.1f} ms")

print("\n== Appendix A.4.1: learning curves vs tuned reservation rule")
res = REF["nominal"]["reservation"]["cost"]
for k, c in S["nominal"]["curves"].items():
    m = np.array(c["mean"])
    ts = np.array(c["timesteps"])
    below = ts[m <= res]
    first = int(below[0]) if len(below) else None
    after = m[ts >= first] if first else []
    print(f"  {k:15s} first <= rule at {first}  max above rule afterwards={max(after - res) if len(after) else float('nan'):+.4f}  "
          f"final={m[-1]:.3f}  range after 20k=[{m[ts >= 20000].min():.3f}, {m[ts >= 20000].max():.3f}]")

print("\n== Sec. 5.5: seeds where nominal weighted PPO beats the unchanged reservation rule")
for s, r in S["shifts"]["rows"].items():
    costs = np.array(r["ppo_weighted_nominal"]["costs"])
    print(f"  {s:40s} {int((costs < r['reservation_nominal']['cost']).sum())}/10  "
          f"thresholds nominal->retuned: {r['reservation_retuned']['thresholds']}")

print("\n== Unrounded values for numbers that sit on a rounding boundary")
opt = REF["nominal"]["optimum"]["cost"]
print(f"  DQN weighted gap   {S['nominal']['rows']['dqn|weighted']['gap_pct']:.4f}%")
print(f"  DQN flat gap       {S['nominal']['rows']['dqn|flat']['gap_pct']:.4f}%")
print(f"  A2C W-F cost diff  {[t['diff'] for t in S['nominal']['tests'] if t['algo'] == 'a2c' and t['metric'] == 'cost'][0]:.5f}")
print(f"  Hetero3 corrected  {S['systems']['rows']['hetero3|corrected']['gap_pct']:.4f}%")
print(f"  Hetero3 corrected, nominal setting cost {S['systems_nominal_setting']['rows']['hetero3|corrected']['cost']['mean']:.5f}")
for s, r in S["shifts"]["rows"].items():
    o = r["optimum"]["cost"]
    g = lambda c: 100 * (c / o - 1)
    print(f"  {s:40s} opt={o:.5f}  W-FT={g(r['ppo_weighted_finetuned']['cost']['mean']):.3f}%  "
          f"F-FT={g(r['ppo_flat_finetuned']['cost']['mean']):.3f}%")
