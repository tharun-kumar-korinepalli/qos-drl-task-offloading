"""Fills table placeholders in the paper from the result files, so every table number comes from the data."""
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
S = REPO / "results" / "summary"
PAPER = Path(__file__).resolve().parent.parent


def pct(x):
    return f"{x:.1f}\\%"


def big(n):
    return f"{n:,}"


def rows_scaling():
    d = json.loads((S / "dp_scaling.json").read_text())
    out = []
    for r in d:
        out.append(f"{r['edge_nodes']} & {big(r['states'])} & {big(r['transition_entries'])} & "
                   f"{r['solve_s']:.2f} & {r['evaluate_s']:.2f} & {r['peak_rss_mb']:.0f} \\\\")
    return "\n".join(out)


def rows_dprewards():
    ref = json.loads((S / "scenario_reference.json").read_text())
    names = {"nominal": "Nominal", "hetero3": "Hetero3", "hetero3_3c": "Hetero3-3C", "large4_3c": "Large4-3C"}
    out = []
    for n, lab in names.items():
        opt = ref[n]["optimum"]["cost"]
        cells = [f"{opt:.4f}"]
        for key in ["weighted", "corrected", "flat", "weighted_gamma0.99", "corrected_gamma0.99", "flat_gamma0.99"]:
            c = ref[n]["dp_rewards"][key]["cost"]
            cells.append(f"{100 * (c / opt - 1):.1f}")
        out.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    return "\n".join(out)


def rows_ratios():
    d = json.loads((S / "reward_analysis.json").read_text())["ratios"]
    out = []
    for r, v in d.items():
        g = lambda k: 100 * (v[k] / v["J_opt"] - 1)
        out.append(f"{r} & {v['J_opt']:.4f} & ${v['wc_corr']:.2f}$, ${v['wa_corr']:.2f}$ & {g('J_corrected'):.1f}\\% & "
                   f"{g('J_matched'):.1f}\\% & {g('J_flat'):.1f}\\% \\\\")
    return "\n".join(out)


def chosen(algo):
    t = json.loads((S / "tuning_selection.json").read_text())
    cfg = dict((k, v) for k, v in t[f"{algo}|weighted"]["config"])
    assert t[f"{algo}|weighted"]["config"] == t[f"{algo}|flat"]["config"]
    if not cfg:
        return "conference setting"
    names = {"learning_rate": "lr", "n_steps": "steps", "batch_size": "batch", "ent_coef": "entropy", "gamma": "$\\gamma$",
             "net_arch": "layers", "target_update_interval": "target", "exploration_fraction": "expl.",
             "buffer_size": "buffer", "reward_scale": "scale", "normalize_advantage": "norm. adv."}
    parts = []
    for k in ["learning_rate", "n_steps", "batch_size", "ent_coef", "gamma", "target_update_interval",
              "exploration_fraction", "buffer_size", "reward_scale", "normalize_advantage", "net_arch"]:
        if k in cfg:
            v = cfg[k]
            if k == "learning_rate":
                v = {1e-4: "$10^{-4}$", 3e-4: "$3\\times10^{-4}$", 7e-4: "$7\\times10^{-4}$", 1e-3: "$10^{-3}$"}[v]
            elif k == "net_arch":
                v = f"$2\\times{v[0]}$"
            elif k == "buffer_size":
                v = f"$10^{len(str(int(v))) - 1}$"
            elif k == "normalize_advantage":
                v = "yes" if v else "no"
            parts.append(f"{names[k]} {v}")
    return "\\shortstack[l]{" + "\\\\".join([", ".join(parts[i:i + 2]) for i in range(0, len(parts), 2)]) + "}"


EXTRA = {}
try:
    sys.path.insert(0, str(PAPER / "gen"))
    from fill_results import EXTRA as E2
    EXTRA.update(E2)
except ImportError:
    pass


def main():
    reps = {"SCALINGROWS": rows_scaling, "DPREWARDROWS": rows_dprewards, "RATIOROWS": rows_ratios,
            "PPOCHOSEN": lambda: chosen("ppo"), "DQNCHOSEN": lambda: chosen("dqn"),
            "A2CCHOSEN": lambda: chosen("a2c"), **EXTRA}
    for src in sorted((PAPER / "parts").glob("part_*.tex")):
        s = src.read_text()
        for k, f in reps.items():
            if k in s:
                s = s.replace(k, f())
        name = "abstract.tex" if src.name == "part_abstract.tex" else src.name
        (PAPER / "build" / name).write_text(s)
    left = set()
    for f in (PAPER / "build").glob("part_*.tex"):
        left |= set(re.findall(r"\b[A-Z0-9]{6,}ROWS\b|\b[A-Z0-9]+CHOSEN\b|\bRES[A-Z0-9]+\b", f.read_text()))
    print("unfilled:", sorted(left))


if __name__ == "__main__":
    (PAPER / "build").mkdir(exist_ok=True)
    main()
