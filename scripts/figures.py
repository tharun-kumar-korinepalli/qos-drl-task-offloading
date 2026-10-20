"""Draw the data figures of the paper from the result files.

Fig. 3 (fig_frontier.pdf), Fig. 4 (fig_systems.pdf) and Fig. 5 (fig_curves.pdf) are drawn here.
Fig. 1 and Fig. 2 are TikZ drawings in figures/architecture.tex and figures/fig_overview.tex.
All figures use Computer Modern and are drawn at their printed width (one column 216.24 pt,
full page 455.24 pt), so they are included without scaling.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from qos_offloading import study as V
from qos_offloading.training import load as load_run

SUM = ROOT / "results" / "summary"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({"text.usetex": False, "font.family": "serif", "font.serif": ["cmr10"],
                     "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
                     "font.size": 7, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5,
                     "ytick.minor.size": 1.5, "pdf.fonttype": 42, "ps.fonttype": 42})
COL = {"weighted": "#b03a2e", "corrected": "#b8860b", "flat": "#3a78b8"}
GRAY = "#555555"
COLW, PAGEW = 216.24 / 72, 455.24 / 72   # printed widths in inches


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)


def load():
    s = json.loads((SUM / "summary.json").read_text())
    s["reference"] = json.loads((SUM / "references.json").read_text())
    return s


# ---------------------------------------------------------------- Fig. 5
def fig_curves(S):
    cur = S["nominal"]["curves"]
    ref = S["reference"]["nominal"]
    ls = {"weighted": "-", "corrected": (0, (4.5, 1.8)), "flat": (0, (1.1, 1.6))}
    fig = plt.figure(figsize=(PAGEW, 140 / 72))
    L, R, B, T, G = 0.075, 0.983, 0.335, 0.925, 0.03
    pw = (R - L - 2 * G) / 3
    for i, (algo, title) in enumerate((("ppo", "(a) PPO"), ("dqn", "(b) DQN"), ("a2c", "(c) A2C"))):
        ax = fig.add_axes([L + i * (pw + G), B, pw, T - B])
        ax.set_yscale("log")                       # every curve fits, from 0.26 to 2.9
        ax.set_ylim(0.22, 3.2)
        ax.set_xlim(0, 100)
        ax.axhline(ref["optimum"]["cost"], color="k", lw=0.8, zorder=2)
        ax.axhline(ref["reservation"]["cost"], color=GRAY, lw=0.8, ls=(0, (4, 2)), zorder=2)
        ax.axhline(ref["dp_rewards"]["flat"]["cost"], color=GRAY, lw=0.8, ls=(0, (1, 1.5)), zorder=2)
        for rw in ("weighted", "corrected", "flat"):
            c = cur.get(f"{algo}|{rw}")
            if c is None:
                continue
            x = np.array(c["timesteps"]) / 1000
            ax.fill_between(x, c["lo"], c["hi"], color=COL[rw], alpha=0.16, lw=0, zorder=1)
            ax.plot(x, c["mean"], color=COL[rw], ls=ls[rw], lw=1.2, zorder=3)
        ax.yaxis.set_major_locator(FixedLocator([0.25, 0.5, 1, 2]))
        ax.yaxis.set_minor_locator(FixedLocator([0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 3]))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_yticklabels(["0.25", "0.5", "1", "2"] if i == 0 else [])
        ax.set_xticks(range(0, 101, 20))
        ax.set_title(title, fontsize=8, pad=3)
        clean(ax)
        if i == 0:
            ax.set_ylabel("SLA cost $J$")
    fig.text(L + (R - L) / 2, 0.185, "training steps (thousands)", ha="center", va="bottom", fontsize=8)
    h = [Line2D([], [], color=COL[r], ls=ls[r], lw=1.2, label=f"{r} reward") for r in ("weighted", "corrected", "flat")]
    h += [Line2D([], [], color="k", lw=0.8, label="optimum (DP)"),
          Line2D([], [], color=GRAY, lw=0.8, ls=(0, (4, 2)), label="tuned reservation rule"),
          Line2D([], [], color=GRAY, lw=0.8, ls=(0, (1, 1.5)), label="DP policy of the flat reward")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(L + (R - L) / 2, -0.015),
               handlelength=2.1, columnspacing=2.0, handletextpad=0.4)
    fig.savefig(OUT / "fig_curves.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 3
def fig_frontier(S):
    ref = S["reference"]["nominal"]
    jobs = V.main_jobs(V.select_configs(load_run))
    pts = {}
    for algo, rws in (("ppo", ("weighted", "corrected", "flat")), ("dqn", ("weighted", "flat"))):
        for rw in rws:
            p = [load_run(jobs[(algo, rw, s)])["exact"]["miss"] for s in V.SEEDS]
            pts[(algo, rw)] = 100 * np.array([m[:2] for m in p])
    fr = 100 * np.array([[c, a] for _, c, a in ref["frontier"]])
    fr = fr[np.argsort(fr[:, 0], kind="stable")]
    pc = lambda key: 100 * np.array(ref[key]["miss"][:2])
    dp = {k: 100 * np.array(ref["dp_rewards"][k]["miss"][:2]) for k in ("weighted", "corrected", "flat")}
    opt = pc("optimum")
    XM, YM = 22, 27
    zx, zy = (1.93, 3.47), (13.75, 16.95)        # crowded region near the optimum
    # the shaded region: everything below the frontier, plus everything left of its smallest critical rate
    poly = np.vstack([[0, 0], [0, YM + 5], [fr[0, 0], YM + 5], fr[fr[:, 1] <= YM + 5], [XM + 1, fr[-1, 1]], [XM + 1, 0]])

    def draw(a, big=False):
        m = 1.7 if big else 1.0
        a.fill(poly[:, 0], poly[:, 1], color="#e6e6e6", lw=0, zorder=0)
        a.plot(fr[:, 0], fr[:, 1], color="k", lw=1.1, zorder=2, solid_joinstyle="round")
        for rw in ("weighted", "corrected", "flat"):
            p = pts[("ppo", rw)]
            a.scatter(p[:, 0], p[:, 1], s=11 * m, color=COL[rw], edgecolor="white", linewidth=0.35,
                      clip_on=big, zorder=5)
        for rw in ("weighted", "flat"):
            p = pts[("dqn", rw)]
            a.scatter(p[:, 0], p[:, 1], s=13 * m, marker="s", facecolor="none", edgecolor=COL[rw], linewidth=0.75,
                      clip_on=big, zorder=5)
        for rw in ("weighted", "corrected", "flat"):       # outlined star in the colour of the reward
            a.scatter(*dp[rw], s=70 * (1.25 if big else 1), marker="*", facecolor="white", edgecolor=COL[rw],
                      linewidth=0.9, clip_on=big, zorder=6)
        a.scatter(*opt, s=26 * (1.25 if big else 1), marker="*", color="k", linewidth=0, clip_on=big, zorder=7)

    fig = plt.figure(figsize=(COLW, 146 / 72))
    ax = fig.add_axes([0.125, 0.185, 0.86, 0.805])
    draw(ax)
    ax.text(0.45, 1.3, "not reachable", color=GRAY, fontsize=7, zorder=3)
    sla, res = pc("sla_rule"), pc("reservation")
    ax.scatter(sla[0], 0, s=26, marker="D", color=GRAY, clip_on=False, zorder=6)
    ax.scatter(*res, s=34, marker="P", color=GRAY, linewidth=0, clip_on=False, zorder=6)
    ax.annotate("SLA rule", xy=(sla[0] + 0.15, 0.5), xytext=(13.4, 1.4), fontsize=7, color="#333333", va="center",
                arrowprops=dict(arrowstyle="-", color="#777777", lw=0.5, shrinkA=1, shrinkB=0))
    ax.text(res[0] + 0.55, res[1], "reservation rule", fontsize=7, color="#333333", va="center")
    ax.set_xlim(0, XM)
    ax.set_ylim(0, YM)
    ax.set_xticks(range(0, 21, 5))
    ax.set_yticks(range(0, 26, 5))
    ax.set_xlabel("critical tasks missed (%)", labelpad=2)
    ax.set_ylabel("analytics tasks missed (%)")
    clean(ax)
    h = [Line2D([], [], color="k", lw=1.1, label="optimal trade-off (DP)"),
         Line2D([], [], marker="*", ls="", color="k", ms=5.5, label="optimum"),
         Line2D([], [], marker="*", ls="", mfc="white", mec="#333333", mew=0.8, ms=8, label="DP policy of a reward"),
         Line2D([], [], marker="o", ls="", color=COL["weighted"], ms=3.6, label="PPO weighted"),
         Line2D([], [], marker="o", ls="", color=COL["corrected"], ms=3.6, label="PPO corrected"),
         Line2D([], [], marker="o", ls="", color=COL["flat"], ms=3.6, label="PPO flat"),
         Line2D([], [], marker="s", ls="", mfc="none", mec="#333333", mew=0.75, ms=3.8, label="DQN (hollow)")]
    lg = ax.legend(handles=h, loc="upper right", frameon=False, handlelength=1.6, handletextpad=0.5,
                   labelspacing=0.22, borderaxespad=0.05)
    # zoom panel, placed below the legend
    fig.canvas.draw()
    top = lg.get_window_extent().transformed(ax.transData.inverted()).y0 - 2.0
    bot = 5.1
    ax.add_patch(plt.Rectangle((zx[0], zy[0]), zx[1] - zx[0], zy[1] - zy[0], fill=False, ec="#666666", lw=0.5,
                               zorder=8))
    az = ax.inset_axes([9.0, bot, 12.6, top - bot], transform=ax.transData)
    draw(az, big=True)
    az.set_xlim(*zx)
    az.set_ylim(*zy)
    az.set_xticks([2.0, 2.5, 3.0])
    az.set_yticks([14, 15, 16])
    az.tick_params(labelsize=6, length=2, width=0.5, pad=1.5)
    for sp in az.spines.values():
        sp.set_linewidth(0.5)
        sp.set_color("#666666")
    # agents that learned the same policy share one point: label how many
    allp = np.vstack(list(pts.values()))
    cnt = np.array([(np.hypot(*(allp - q).T) < 0.01).sum() for q in allp])
    g = allp[np.argmax(cnt)]
    az.annotate(f"{cnt.max()} agents", xy=g, xytext=(2.80, 16.2), fontsize=6.5, color="#333333", va="center",
                arrowprops=dict(arrowstyle="-", color="#777777", lw=0.5, shrinkA=1, shrinkB=3))
    ax.text(9.0, top + 0.35, "zoom of the boxed area", fontsize=6.5, color=GRAY, va="bottom")
    fig.savefig(OUT / "fig_frontier.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 4
SYS_NAME = {"nominal": "Nominal", "hetero3": "Hetero3", "hetero3_3c": "Hetero3-3C", "large4_3c": "Large4-3C"}


def fig_systems(S):
    rows = S["systems"]["rows"]
    ref = S["reference"]
    items = [("SLA rule", "sla_rule"), ("reservation rule", "reservation"), ("node-set rule", "node_sets"),
             ("PPO flat", "flat"), ("PPO weighted", "weighted"), ("PPO corrected", "corrected")]
    fig = plt.figure(figsize=(PAGEW, 102 / 72))
    L, R, B, T, G = 0.128, 0.982, 0.245, 0.87, 0.036
    pw = (R - L - 3 * G) / 4
    minor = [v * d for d in (10, 100) for v in range(2, 10)] + [2, 4, 6, 8]
    for i, sysname in enumerate(SYS_NAME):
        ax = fig.add_axes([L + i * (pw + G), B, pw, T - B])
        ax.set_xscale("symlog", linthresh=10, linscale=1, base=10)   # linear below 10 %, log above
        ax.set_xlim(0, 1000)
        ax.set_ylim(5.6, -0.6)
        ax.xaxis.set_major_locator(FixedLocator([0, 10, 100, 1000]))
        ax.xaxis.set_minor_locator(FixedLocator(minor))
        ax.set_xticklabels(["0", "10", "100", "1000"])
        for v in (10, 100, 1000):
            ax.axvline(v, color="#e3e3e3", lw=0.5, zorder=0)
        ax.axhline(2.5, color="#cfcfcf", lw=0.5, zorder=0)
        opt = ref[sysname]["optimum"]["cost"]
        for k, (lab, key) in enumerate(items):
            if k < 3:
                ax.scatter(100 * (ref[sysname][key]["cost"] / opt - 1), k, s=20, marker="D", color=GRAY,
                           linewidth=0, zorder=4, clip_on=False)
                continue
            src = S["nominal"]["rows"][f"ppo|{key}"] if sysname == "nominal" else rows[f"{sysname}|{key}"]
            costs = 100 * (np.array(src["costs"]) / opt - 1)
            m, lo, hi = (100 * (src["cost"][q] / opt - 1) for q in ("mean", "ci_low", "ci_high"))
            ax.scatter(costs, k + np.linspace(-0.2, 0.2, len(costs)), s=6, color=COL[key], alpha=0.85, linewidth=0,
                       zorder=3, clip_on=False)
            ax.plot([lo, hi], [k, k], color="k", lw=0.8, zorder=5, solid_capstyle="butt")
            ax.plot([m, m], [k - 0.27, k + 0.27], color="k", lw=1.0, zorder=5, solid_capstyle="butt")
        ax.set_yticks(range(len(items)))
        ax.set_yticklabels([lab for lab, _ in items] if i == 0 else [])
        ax.tick_params(axis="y", length=0, pad=3)
        ax.set_title(SYS_NAME[sysname], fontsize=8, pad=3)
        clean(ax)
    fig.text(L + (R - L) / 2, 0.02, "above the optimum of the system (%)", ha="center", va="bottom", fontsize=8)
    fig.savefig(OUT / "fig_systems.pdf")
    plt.close(fig)


if __name__ == "__main__":
    S = load()
    fig_curves(S)
    fig_frontier(S)
    fig_systems(S)
