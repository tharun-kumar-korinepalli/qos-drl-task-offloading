import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from qos_offloading import studies as S
from qos_offloading.runner import load

HERE = Path(__file__).resolve().parent
SUMMARY = HERE.parent / "results" / "summary" / "summary.json"
OUT = Path(os.environ.get("FIG_DIR", HERE))

plt.rcParams.update({"font.family": ["TeX Gyre Termes", "Liberation Serif", "DejaVu Serif"], "font.size": 7,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5, "legend.frameon": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
CRIT, ANA = "#8b1a1a", "#4a7fb5"
W_COL, F_COL = "#8b1a1a", "#4a7fb5"
ALGO_MARK = {"ppo": "o", "dqn": "s", "a2c": "^"}
ALGO_NAME = {"ppo": "PPO", "dqn": "DQN", "a2c": "A2C"}
COL1, COL2 = 3.45, 7.16


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)


def m_of(d, key):
    v = d[key]
    return (v["mean"], v["ci_low"], v["ci_high"]) if isinstance(v, dict) else (v, v, v)


def fig_policies(res):
    s = res["main"]["summary"]
    order = ["Optimal SLA cost (DP)", "Optimal for weighted reward (DP)", "Optimal for flat reward (DP)",
             "PPO weighted", "PPO flat", "DQN weighted", "DQN flat", "A2C weighted", "A2C flat",
             "Reservation rule (tuned)", "SLA rule", "JSQ", "RR"]
    labels = {"Optimal SLA cost (DP)": "Optimum (DP)", "Optimal for weighted reward (DP)": "DP, weighted reward",
              "Optimal for flat reward (DP)": "DP, flat reward", "Reservation rule (tuned)": "Reservation rule"}
    order = [o for o in order if o in s]
    fig, ax = plt.subplots(figsize=(COL1, 2.55))
    y = np.arange(len(order))[::-1]
    h = 0.38
    for yi, name in zip(y, order):
        for off, key, col in [(h / 2, "class_0_violation_rate", CRIT), (-h / 2, "class_1_violation_rate", ANA)]:
            mu, lo, hi = (100 * v for v in m_of(s[name], key))
            ax.barh(yi + off, mu, h, color=col, lw=0)
            if hi > lo:
                ax.errorbar(mu, yi + off, xerr=[[mu - lo], [hi - mu]], fmt="none", ecolor="k", elinewidth=0.5,
                            capsize=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels([labels.get(o, o) for o in order])
    ax.set_xlabel("tasks that miss their latency limit (%)")
    ax.legend(handles=[Line2D([], [], color=CRIT, lw=4, label="critical"),
                       Line2D([], [], color=ANA, lw=4, label="analytics")], loc="lower right", fontsize=6)
    ax.axhline(y[2] - 0.5, color="0.6", lw=0.4, ls=":")
    ax.axhline(y[8] - 0.5, color="0.6", lw=0.4, ls=":")
    clean(ax)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_policies.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def pareto(points):
    pts = sorted({(round(c, 6), round(a, 6)) for _, c, a in points})
    front, best = [], np.inf
    for c, a in pts:
        if a < best - 1e-9:
            front.append((c, a))
            best = a
    return np.array(front)


def fig_frontier(res):
    ref = res["reference"]
    fr = pareto(ref["frontier"]) * 100
    fig, ax = plt.subplots(figsize=(COL1, 2.4))
    ax.plot(fr[:, 0], fr[:, 1], color="k", lw=0.9, zorder=2, label="optimal trade-off (DP)")
    ax.fill_between(fr[:, 0], fr[:, 1], 0, color="0.92", zorder=0, lw=0)
    for algo in ["ppo", "dqn"]:
        for rw, col, fc in [("weighted", W_COL, W_COL), ("flat", F_COL, "white")]:
            xs, ys = [], []
            for seed in S.SEEDS:
                r = load(S.main_jobs()[(algo, rw, seed)])
                if r:
                    xs.append(100 * r["exact"]["class_0_violation_rate"])
                    ys.append(100 * r["exact"]["class_1_violation_rate"])
            ax.scatter(xs, ys, s=9, marker=ALGO_MARK[algo], facecolors=fc, edgecolors=col, linewidths=0.6, zorder=3)
    long_pts = []
    for key, jobs in [("ppo", S.curve_jobs()), ("dqn-scaled", S.dqn_scaled_jobs())]:
        for k, j in jobs.items():
            if (key == "ppo" and k[:2] == ("ppo", "weighted")) or (key == "dqn-scaled" and k[0] == "weighted"):
                r = load(j)
                if r:
                    long_pts.append((key, 100 * r["exact"]["class_0_violation_rate"],
                                     100 * r["exact"]["class_1_violation_rate"]))
    for key, mk in [("ppo", "o"), ("dqn-scaled", "s")]:
        pts = [(x, y) for k, x, y in long_pts if k == key]
        if pts:
            ax.scatter(*zip(*pts), s=11, marker=mk, facecolors="#e8a33d", edgecolors="k", linewidths=0.4, zorder=4)
    for key, mk, lab, off in [("threshold", "P", "SLA rule", (2, 10)), ("reservation", "X", "reservation rule", (4, -2))]:
        ax.scatter(100 * ref[key]["class_0_violation_rate"], 100 * ref[key]["class_1_violation_rate"], marker=mk,
                   s=22, color="0.25", zorder=4)
        ax.annotate(lab, (100 * ref[key]["class_0_violation_rate"], 100 * ref[key]["class_1_violation_rate"]),
                    xytext=off, textcoords="offset points", fontsize=6, color="0.25")
    for key, col, lab in [("dp_reward_weighted", W_COL, "DP, weighted"), ("dp_reward_flat", F_COL, "DP, flat")]:
        ax.scatter(100 * ref[key]["class_0_violation_rate"], 100 * ref[key]["class_1_violation_rate"], marker="*",
                   s=55, color=col, edgecolors="k", linewidths=0.4, zorder=5)
        ax.annotate(lab, (100 * ref[key]["class_0_violation_rate"], 100 * ref[key]["class_1_violation_rate"]),
                    xytext=(-36, -12) if "weighted" in key else (-30, 6), textcoords="offset points", fontsize=6,
                    color=col)
    ax.text(1.0, 1.0, "not achievable", fontsize=6, color="0.45", style="italic")
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 26)
    ax.set_xlabel("critical tasks missed (%)")
    ax.set_ylabel("analytics tasks missed (%)")
    handles = [Line2D([], [], color="k", lw=0.9, label="optimal trade-off (DP)")]
    handles += [Line2D([], [], ls="", marker=ALGO_MARK[a], mfc="0.5", mec="0.5", ms=3.5, label=ALGO_NAME[a])
                for a in ["ppo", "dqn"]]
    handles += [Line2D([], [], ls="", marker="o", mfc=W_COL, mec=W_COL, ms=3.5, label="weighted reward"),
                Line2D([], [], ls="", marker="o", mfc="white", mec=F_COL, ms=3.5, label="flat reward"),
                Line2D([], [], ls="", marker="o", mfc="#e8a33d", mec="k", ms=3.5, label="weighted, 100k steps")]
    ax.legend(handles=handles, fontsize=5.5, loc="upper right", ncol=2, handletextpad=0.3, columnspacing=0.8)
    clean(ax)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_frontier.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def errbar(ax, x, d, key, col, mk, label, mfc=None):
    mu = np.array([100 * r[key]["mean"] if r and key in r else np.nan for r in d])
    lo = np.array([100 * r[key]["ci_low"] if r and key in r else np.nan for r in d])
    hi = np.array([100 * r[key]["ci_high"] if r and key in r else np.nan for r in d])
    ax.errorbar(x, mu, yerr=[mu - lo, hi - mu], color=col, marker=mk, ms=3, lw=0.9, capsize=1.3, elinewidth=0.5,
                mfc=mfc or col, label=label)


def fig_sweeps(res):
    fig, axs = plt.subplots(1, 3, figsize=(COL2, 1.75))
    rows = res["penalty"]["rows"]
    x = np.arange(len(rows))
    errbar(axs[0], x, rows, "class_0_violation_rate", CRIT, "o", "critical, PPO")
    errbar(axs[0], x, rows, "class_1_violation_rate", ANA, "s", "analytics, PPO")
    axs[0].plot(x, [100 * r["dp_reward_optimal"]["class_0_violation_rate"] for r in rows], color=CRIT, ls="--",
                lw=0.7, label="critical, DP")
    axs[0].plot(x, [100 * r["dp_reward_optimal"]["class_1_violation_rate"] for r in rows], color=ANA, ls="--",
                lw=0.7, label="analytics, DP")
    axs[0].set_xticks(x)
    axs[0].set_xticklabels([f"{r['w_crit']:g}:{r['w_ana']:g}" for r in rows])
    axs[0].set_xlabel("penalty critical : analytics (mean 6)")
    axs[0].set_ylabel("tasks missed (%)")
    axs[0].set_title("(a) Penalty ratio", fontsize=7)
    axs[0].legend(fontsize=5.5, ncol=2, loc="upper center", columnspacing=0.8)
    axs[0].set_ylim(0, 27)

    rows = res["load"]["rows"]
    xs = [r["utilization"] for r in rows]
    for rw, col, mfc in [("weighted", W_COL, W_COL), ("flat", F_COL, "white")]:
        d = [r[rw] for r in rows]
        mu = np.array([r["sla_cost_per_task"]["mean"] if r else np.nan for r in d])
        lo = np.array([r["sla_cost_per_task"]["ci_low"] if r else np.nan for r in d])
        hi = np.array([r["sla_cost_per_task"]["ci_high"] if r else np.nan for r in d])
        axs[1].errorbar(xs, mu, yerr=[mu - lo, hi - mu], color=col, marker="o", ms=3, lw=0.9, capsize=1.3,
                        elinewidth=0.5, mfc=mfc, label=f"PPO, {rw}")
        key = "dp_reward_weighted" if rw == "weighted" else "dp_reward_flat"
        axs[1].plot(xs, [r["reference"][key]["sla_cost_per_task"] for r in rows], color=col, ls="--", lw=0.7,
                    label=f"DP, {rw}")
    axs[1].plot(xs, [r["reference"]["dp_sla_optimal"]["sla_cost_per_task"] for r in rows], color="k", lw=0.7,
                ls=":", label="optimum")
    axs[1].set_xlabel("offered edge load $\\rho$")
    axs[1].set_ylabel("SLA cost per task")
    axs[1].set_title("(b) Edge load", fontsize=7)
    axs[1].legend(fontsize=5.5, loc="upper left")

    rows = res["mix"]["rows"]
    xs = [100 * r["critical_prob"] for r in rows]
    for rw, col, mfc in [("weighted", W_COL, W_COL), ("flat", F_COL, "white")]:
        d = [r[rw] for r in rows]
        mu = np.array([100 * r["class_0_violation_rate"]["mean"] if r else np.nan for r in d])
        lo = np.array([100 * r["class_0_violation_rate"]["ci_low"] if r else np.nan for r in d])
        hi = np.array([100 * r["class_0_violation_rate"]["ci_high"] if r else np.nan for r in d])
        axs[2].errorbar(xs, mu, yerr=[mu - lo, hi - mu], color=col, marker="o", ms=3, lw=0.9, capsize=1.3,
                        elinewidth=0.5, mfc=mfc, label=f"PPO, {rw}")
        key = "dp_reward_weighted" if rw == "weighted" else "dp_reward_flat"
        axs[2].plot(xs, [100 * r["reference"][key]["class_0_violation_rate"] for r in rows], color=col, ls="--",
                    lw=0.7, label=f"DP, {rw}")
    axs[2].set_xlabel("share of critical tasks (%)")
    axs[2].set_ylabel("critical tasks missed (%)")
    axs[2].set_title("(c) Class mix", fontsize=7)
    axs[2].legend(fontsize=5.5, loc="upper right", ncol=2, columnspacing=0.8)
    axs[2].set_ylim(0, 40)
    for ax in axs:
        clean(ax)
        ax.set_ylim(bottom=0)
    fig.tight_layout(pad=0.3, w_pad=1.2)
    fig.savefig(OUT / "fig_sweeps.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def fig_curves(res):
    cur = res["curves"]["curves"]
    ref = res["reference"]
    panels = [(a, n) for a, n in [("ppo", "PPO"), ("dqn", "DQN"), ("dqn-scaled", "DQN, rewards x 0.1")]
              if f"{a}_weighted" in cur]
    fig, axs = plt.subplots(1, len(panels), figsize=(COL2, 1.75), sharey=True)
    for i, (ax, (algo, title)) in enumerate(zip(np.atleast_1d(axs), panels)):
        for rw, col in [("weighted", W_COL), ("flat", F_COL)]:
            c = cur.get(f"{algo}_{rw}")
            if not c:
                continue
            t = np.array(c["timesteps"]) / 1000
            mu = np.array(c["sla_cost_per_task"]["mean"])
            ci = np.array(c["sla_cost_per_task"]["ci"])
            ax.plot(t, mu, color=col, lw=0.9, label=f"{rw} reward")
            ax.fill_between(t, ci[:, 0], ci[:, 1], color=col, alpha=0.18, lw=0)
        ax.axhline(ref["dp_sla_optimal"]["sla_cost_per_task"], color="k", lw=0.6, ls=":", label="optimum (DP)")
        ax.axhline(ref["dp_reward_flat"]["sla_cost_per_task"], color=F_COL, lw=0.6, ls="--",
                   label="optimum for flat reward (DP)")
        ax.axhline(ref["reservation"]["sla_cost_per_task"], color="0.4", lw=0.6, ls="-.", label="reservation rule")
        ax.axvline(20, color="0.8", lw=0.5)
        ax.set_title(f"({'abc'[i]}) {title}", fontsize=7)
        ax.set_xlabel("training steps (thousands)")
        clean(ax)
    axs[0].set_ylabel("SLA cost per task")
    axs[0].set_ylim(0, 1.2)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=6, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(pad=0.3, w_pad=0.8, rect=(0, 0.1, 1, 1))
    fig.savefig(OUT / "fig_curves.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def main():
    res = json.loads(SUMMARY.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    fig_policies(res)
    fig_frontier(res)
    fig_sweeps(res)
    if res["curves"]["curves"]:
        fig_curves(res)
    print(f"figures written to {OUT}")


if __name__ == "__main__":
    main()
