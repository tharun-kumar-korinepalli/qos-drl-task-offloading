"""Overview of the evaluation method (Fig. 2)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({"font.family": "TeX Gyre Termes", "mathtext.fontset": "stix"})
fig, ax = plt.subplots(figsize=(7.16, 2.45))
ax.set_xlim(0, 200)
ax.set_ylim(0, 68)
ax.axis("off")

C_SYS = ("#f2f2f2", "#888888")
C_DP = ("#e9eef7", "#3a78b8")
C_RL = ("#f6e9e7", "#b03a2e")
C_RULE = ("#efefe6", "#7a7a52")
C_OUT = ("#ffffff", "#222222")


def box(x, y, w, h, title, lines, col, tsize=7.4, lsize=6.2):
    fc, ec = col
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.6", fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h - 3.2, title, ha="center", va="center", fontsize=tsize, weight="bold")
    ax.text(x + w / 2, y + (h - 5.2) / 2, lines, ha="center", va="center", fontsize=lsize, color="#333333",
            linespacing=1.25)


def arrow(p, q, col="#222222", ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.8, color=col, linestyle=ls,
                                 shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}"))


def tag(x, y, s):
    ax.text(x, y, s, fontsize=5.8, color="#555555", ha="center", va="center", style="italic")


# system
box(2, 22, 34, 26, "Offloading system", "$N$ edge nodes, $K$ classes\nqueues $q_i \\leq 8$, limits $L_c$\n"
    "random background, cloud\nmeets $L_c$ w.p. $\\pi_c$", C_SYS, 7.4, 5.9)
# exact model
box(48, 40, 46, 24, "Exact model", "state $(q_1,\\dots,q_N,c,\\tau)$\n$9^N K T$ states, sparse $P(s'|s,a)$\n"
    "backward induction (DP)", C_DP)
# rl
box(48, 4, 46, 26, "Deep RL training", "PPO, DQN, A2C, 10 seeds\nreward: weighted, corrected, flat\n"
    "tuned by exact expected return", C_RL)
# rules
box(103, 24, 41, 16, "Rules", "JSQ, SLA rule, reservation\nthresholds tuned on exact model", C_RULE, 7.0, 6.0)
# outputs of DP
box(103, 46, 41, 18, "Exact references", "optimum $J^\\ast$, Pareto frontier\nDP policy of each reward", C_DP, 7.0, 6.0)
# table
box(103, 4, 41, 16, "Policy table", "greedy action of the network\nin every state, queried once", C_RL, 7.0, 6.0)
# evaluation
box(152, 18, 46, 30, "Exact evaluation", "forward pass of the state distribution\n$v_c$, latency, edge use, $J$, return\n"
    "no test-set noise\nchecked against simulation", C_DP, 7.4, 5.8)
# comparison
ax.text(175, 9.5, "gap to $J^\\ast$, seeds against rules,\nMann-Whitney, $A_{12}$, Holm", ha="center", va="center",
        fontsize=6.2, color="#222222")
arrow((175, 18), (175, 14.2))

arrow((36, 42), (48, 52))
arrow((36, 28), (48, 17))
tag(41, 55.5, "model")
tag(41, 14.5, "simulator")
arrow((94, 55), (103, 55))
arrow((94, 46), (103, 34), col="#7a7a52")
tag(97.5, 37, "tune")
arrow((94, 12), (103, 12))
arrow((144, 55), (152, 42))
arrow((144, 32), (152, 33))
arrow((144, 12), (152, 24))

fig.savefig("fig_overview.pdf", bbox_inches="tight", pad_inches=0.01)
