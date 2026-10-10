import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "TeX Gyre Termes"
plt.rcParams["mathtext.fontset"] = "stix"

fig, ax = plt.subplots(figsize=(3.45, 2.15))
ax.set_xlim(0, 100); ax.set_ylim(0, 66); ax.axis("off")


def box(x, y, w, h, title, sub, fc, ec, tsize=7.8, ssize=6.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.5", fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h - 3.3, title, ha="center", va="center", fontsize=tsize, weight="bold")
    ax.text(x + w / 2, y + (h - 6.2) / 2 + 0.2, sub, ha="center", va="center", fontsize=ssize,
            color="#333333", linespacing=1.15)


def arrow(p, q, ls="-", color="#222222"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.8, color=color,
                                 linestyle=ls, shrinkA=1, shrinkB=1))


def num(x, y, n):
    ax.add_patch(plt.Circle((x, y), 2.4, fc="#222222", ec="none"))
    ax.text(x, y, str(n), color="white", fontsize=6, ha="center", va="center", weight="bold")


box(9, 55, 82, 10.5, "IoT devices", r"one task per step, class $c$ with latency limit $L_c$", "#f2f2f2", "#888888")
box(22, 34, 56, 12, "RL agent (PPO, DQN or A2C)", r"reads state $(q_1,\dots,q_N,c,\tau)$" + "\n" + "chooses an edge node or the cloud", "#ece4f7", "#7b5fa8")
ax.add_patch(FancyBboxPatch((3, 3), 94, 21.5, boxstyle="round,pad=0.3,rounding_size=1.5", fc="none", ec="#888888", lw=0.8, linestyle="--"))
ax.text(50, 21.6, "Environment: edge-cloud continuum", ha="center", va="center", fontsize=7.2, weight="bold")
box(5, 4.5, 28.5, 13, "Edge node 1", r"latency $\ell_{0,1}+\delta_1 q_1$" + "\n" + "queue of up to 8", "#e3f2e6", "#3d8b55", 7.2, 6.0)
box(35.75, 4.5, 28.5, 13, r"Edge nodes 2 to $N$", "own speed and queue" + "\n" + "+ background traffic", "#e3f2e6", "#3d8b55", 7.2, 6.0)
box(66.5, 4.5, 28.5, 13, "Cloud", "no queue, 110 ms" + "\n" + r"meets $L_c$ w.p. $\pi_c$", "#e3f0fa", "#3b78b0", 7.2, 6.0)

arrow((50, 55), (50, 46.5)); num(45, 50.7, 1); ax.text(53.5, 50.7, r"task of class $c$", fontsize=6.2, va="center")
arrow((10, 24.8), (10, 40), ls="--", color="#555555"); arrow((10, 40), (24, 40), ls="--", color="#555555")
num(17, 43.6, 2); ax.text(17, 36.6, "state", fontsize=6.2, ha="center", color="#555555")
arrow((50, 34), (50, 24.8)); num(45, 29.4, 3); ax.text(53.5, 29.4, r"action $a$", fontsize=6.2, va="center")
arrow((90, 24.8), (90, 40), color="#555555"); arrow((90, 40), (76, 40), color="#555555")
num(80, 44.2, 4); ax.text(79.5, 47.8, "latency", fontsize=6.2, ha="center", color="#555555")
num(80, 35.8, 5); ax.text(78.5, 31.6, r"reward $R_t$", fontsize=6.2, ha="center", color="#555555")

fig.savefig("architecture.pdf", bbox_inches="tight", pad_inches=0.01)
