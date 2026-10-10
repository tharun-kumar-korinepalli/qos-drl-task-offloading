"""Result tables of the paper, generated from results/summary/summary_v3.json."""
import json
from pathlib import Path

S = json.loads((Path(__file__).resolve().parents[2] / "results" / "summary" / "summary_v3.json").read_text())
REF = S["reference"]
SYS = ["nominal", "hetero3", "hetero3_3c", "large4_3c"]
SYSN = {"nominal": "Nominal", "hetero3": "Hetero3", "hetero3_3c": "Hetero3-3C", "large4_3c": "Large4-3C"}


def p(x):
    return f"{100 * x:.1f}"


def pv(x):
    if x < 0.001:
        return "$<0.001$"
    return f"{x:.3f}"


def ci(d, f=lambda v: f"{v:.3f}"):
    return f"{f(d['mean'])} [{f(d['ci_low'])}, {f(d['ci_high'])}]"


def main_rows():
    ref = REF["nominal"]
    opt = ref["optimum"]["cost"]
    out = []
    for key, lab in [("jsq", "JSQ"), ("sla_rule", "SLA rule"), ("reservation", "Reservation rule (tuned)")]:
        v = ref[key]
        out.append(f"{lab} & {p(v['miss'][0])} & {p(v['miss'][1])} & {p(v['overall'])} & {v['latency'][0]:.1f} & "
                   f"{v['latency'][1]:.1f} & {p(v['edge'])} & {v['cost']:.3f} & {100 * (v['cost'] / opt - 1):.0f}\\% & \\\\")
    out.append("\\midrule")
    rows = S["main"]["rows"]
    for algo, an in [("a2c", "A2C"), ("dqn", "DQN"), ("ppo", "PPO")]:
        for rw in ["flat", "weighted", "corrected"]:
            k = f"{algo}|{rw}"
            if k not in rows:
                continue
            v = rows[k]
            out.append(f"{an} {rw} & {p(v['miss0']['mean'])} & {p(v['miss1']['mean'])} & {p(v['overall']['mean'])} & "
                       f"{v['lat0']['mean']:.1f} & {v['lat1']['mean']:.1f} & {p(v['edge']['mean'])} & "
                       f"{ci(v['cost'])} & {v['gap_pct']:.0f}\\% & {v['vs_reservation']['better']}/10 \\\\")
    out.append("\\midrule")
    for key, lab in [("flat", "DP flat"), ("weighted", "DP weighted"), ("corrected", "DP corrected")]:
        v = ref["dp_rewards"][key]
        out.append(f"{lab} & {p(v['miss'][0])} & {p(v['miss'][1])} & {p(v['overall'])} & {v['latency'][0]:.1f} & "
                   f"{v['latency'][1]:.1f} & {p(v['edge'])} & {v['cost']:.3f} & {100 * (v['cost'] / opt - 1):.0f}\\% & \\\\")
    v = ref["optimum"]
    out.append(f"Optimum (DP) & {p(v['miss'][0])} & {p(v['miss'][1])} & {p(v['overall'])} & {v['latency'][0]:.1f} & "
               f"{v['latency'][1]:.1f} & {p(v['edge'])} & {v['cost']:.3f} & 0\\% & \\\\")
    return "\n".join(out)


def test_rows():
    out = []
    name = {"cost": "SLA cost", "miss0": "critical misses", "miss1": "analytics misses"}
    last = None
    for t in S["main"]["tests"]:
        grp = (t["algo"], t["a"], t["b"])
        if last is not None and grp[0] != last:
            out.append("\\midrule")
        last = grp[0]
        d = t["diff"] if t["metric"] == "cost" else 100 * t["diff"]
        ds = f"{d:+.3f}" if t["metric"] == "cost" else f"{d:+.1f}"
        out.append(f"{t['algo'].upper()} & {t['a']} vs {t['b']} & {name[t['metric']]} & ${ds}$ & {t['a12']:.2f} & "
                   f"{pv(t['p_holm'])} & {t['lower']}/{t['n']} \\\\")
    return "\n".join(out)


def scen_rows():
    rows = S["scenarios"]["rows"]
    out = []
    for i, sysn in enumerate(SYS[1:]):
        ref = REF[sysn]
        opt = ref["optimum"]["cost"]
        if i:
            out.append("\\addlinespace")
        out.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{{SYSN[sysn]}}}, optimum {opt:.4f}}} \\\\")
        for key, lab in [("sla_rule", "SLA rule"), ("reservation", "Reservation rule"),
                         ("reservation_sets", "Node-set rule")]:
            v = ref[key]
            out.append(f"\\quad {lab} & {v['cost']:.3f} & {100 * (v['cost'] / opt - 1):.0f}\\% & "
                       f"{', '.join(p(x) for x in v['miss'])} & \\\\")
        for rw in ["flat", "weighted", "corrected"]:
            v = rows[f"{sysn}|{rw}"]
            miss = ", ".join(p(v[f"miss{c}"]["mean"]) for c in range(len(ref["optimum"]["miss"])))
            out.append(f"\\quad PPO {rw} & {ci(v['cost'])} & {v['gap_pct']:.0f}\\% & {miss} & "
                       f"{v['vs_reservation_sets']['better']}/10 \\\\")
    return "\n".join(out)


def pub_rows():
    rows = S["published"]["rows"]
    out = []
    lab = {"full": "full design", "adaptive_only": "adaptive tiers only"}
    for k, v in rows.items():
        sysn, var, eta = k.split("|")
        src = "exact" if v["source"] == "exact" else "sim."
        out.append(f"{lab[var]} & {eta} & {ci(v['cost'])} & {v['gap_pct']:.0f}\\% & {p(v['miss0']['mean'])} & "
                   f"{', '.join(f'{x:.1f}' for x in v['final_penalties'])} \\\\")
    return "\n".join(out)


def shift_rows():
    rows = S["robustness"]["rows"]
    out = []
    for s, r in rows.items():
        opt = r["optimum"]["cost"]
        g = lambda c: f"{100 * (c / opt - 1):.0f}\\%"
        cells = [f"{opt:.3f}", g(r["sla_rule_nominal_model"]["cost"]), g(r["reservation_nominal"]["cost"]),
                 g(r["reservation_retuned"]["cost"])]
        for k in ["ppo_weighted_nominal", "ppo_weighted_finetuned", "ppo_flat_nominal", "ppo_flat_finetuned"]:
            cells.append(g(r[k]["cost"]["mean"]) if k in r else "n/a")
        out.append(s.split(" (")[0] + " & " + " & ".join(cells) + " \\\\")
    return "\n".join(out)


def _m(rw, what):
    v = S["main"]["rows"][f"ppo|{rw}"]
    if what == "ci":
        return ci(v["cost"])
    if what == "gap":
        return f"{v['gap_pct']:.0f}\\%"
    return p(v["miss0"]["mean"])


EXTRA = {"PPOWCI": lambda: _m("weighted", "ci"), "PPOWGAP": lambda: _m("weighted", "gap"),
         "PPOWCRIT": lambda: _m("weighted", "crit"), "PPOFCI": lambda: _m("flat", "ci"),
         "PPOFGAP": lambda: _m("flat", "gap"), "PPOFCRIT": lambda: _m("flat", "crit"),"TABMAINROWS": main_rows, "TABTESTROWS": test_rows, "TABSCENROWS": scen_rows, "TABPUBROWS": pub_rows,
         "TABSHIFTROWS": shift_rows}
