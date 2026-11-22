"""Supplementary figures with readable labels (run from plism-paper/)."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

R = "/home/claude/plism-results/results/"
OUT = "figures/"
NAME = {"uni2h": "UNI2-h", "gigapath": "Prov-GigaPath", "hoptimus0": "H-optimus-0", "uni": "UNI",
        "virchow2": "Virchow2", "phikon2": "Phikon-v2", "dinov2_l": "DINOv2-L", "conch": "CONCH",
        "vitb16_in": "ViT-B/16 (ImageNet)"}
ORDER = list(NAME)  # sorted by linear-probe drop, as in Table 1
PHONE = {"iphone13": "iPhone 13 mini", "redmi": "Redmi Note 9 Pro", "moto": "moto g8",
         "iphone6": "iPhone 6", "galaxy": "Galaxy S20 5G", "itel": "itel P33"}
# relative sharpness per phone (median Laplacian variance / scanner median), from the original analysis output
SHARP = {"itel": 0.568, "galaxy": 1.000, "moto": 1.645, "redmi": 2.331, "iphone6": 2.384, "iphone13": 3.176}
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e4e4e4"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "serif", "pdf.fonttype": 42})

def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT + name + ".pdf"); fig.savefig(OUT + name + ".png", dpi=200); plt.close(fig)

# 1. drop vs relative sharpness: per-encoder points (grey) + mean over encoders (blue), phones labelled
d = pd.read_csv(R + "per_phone_drop.csv").set_index("model")
fig, ax = plt.subplots(figsize=(3.4, 2.7))
xs = np.array([SHARP[p] for p in d.columns])
for m in d.index:
    ax.scatter(xs, d.loc[m].values, s=10, color="#bdbdbd", zorder=2, linewidths=0)
mean = d.mean(axis=0)
ax.plot([SHARP[p] for p in sorted(SHARP, key=SHARP.get)], [mean[p] for p in sorted(SHARP, key=SHARP.get)],
        color=CAT[0], lw=1.5, zorder=3)
ax.scatter(xs, mean.values, s=28, color=CAT[0], zorder=4, edgecolors="white", linewidths=1)
off = {"itel": (5, 3, "left"), "galaxy": (5, 4, "left"), "moto": (0, 6, "center"), "redmi": (-3, -22, "right"),
       "iphone6": (0, 7, "center"), "iphone13": (2, -12, "right")}
for p in d.columns:
    dx, dy, ha = off[p]
    ax.annotate(PHONE[p], (SHARP[p], mean[p]), xytext=(dx, dy), textcoords="offset points", ha=ha,
                fontsize=6.5, color=INK)
rho, pv = spearmanr(xs, mean.values)
ax.set_title(f"mean over encoders: Spearman $\\rho$ = {rho:.2f}, p = {pv:.2f}", fontsize=7, color=INK)
ax.set_xscale("log"); ax.set_xticks([0.6, 1, 2, 3]); ax.set_xticklabels(["0.6", "1", "2", "3"])
ax.set_xlabel("relative sharpness (phone / scanner median)"); ax.set_ylabel("linear-probe BA drop")
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.scatter([], [], s=10, color="#bdbdbd", label="single encoder"); ax.scatter([], [], s=28, color=CAT[0], label="mean over encoders")
ax.legend(loc="upper right", frameon=False, fontsize=6.5)
ax.set_ylim(-0.01, 0.34)
save(fig, "fig_quality")

# 2. paired cosine similarity: median and IQR per encoder and comparison
p = pd.read_csv(R + "paired_consistency.csv"); p = p[p.level == "overall"]
comps = ["scanner-scanner", "phone-scanner", "phone-phone", "phone-scanner (different location)"]
labels = ["scanner–scanner", "phone–scanner", "phone–phone", "phone–scanner (other location)"]
fig, ax = plt.subplots(figsize=(3.4, 3.2))
y = np.arange(len(ORDER))
for j, (c, lab) in enumerate(zip(comps, labels)):
    s = p[p.comparison == c].set_index("model").reindex(ORDER)
    yy = y + (j - 1.5) * 0.18
    ax.hlines(yy, s.q25, s.q75, color=CAT[j], lw=1.2)
    ax.scatter(s["median"], yy, s=12, color=CAT[j], label=lab, zorder=3)
ax.set_yticks(y); ax.set_yticklabels([NAME[m] for m in ORDER]); ax.invert_yaxis()
ax.set_xlabel("cosine similarity (median, IQR)"); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(loc="upper center", bbox_to_anchor=(0.3, -0.17), ncol=2, frameon=False, fontsize=6.5, columnspacing=0.8, handletextpad=0.2)
save(fig, "fig_paired")

# 3. retrieval top-k, phone->scanner, same-stain and cross-stain
r = pd.read_csv(R + "retrieval_summary.csv"); r = r[r.kind == "phone->scanner"]
SEQ = ["#86b6ef", "#3987e5", "#1c5cab"]
fig, axes = plt.subplots(1, 2, figsize=(3.4, 3.0), sharey=True)
for ax, prot, title in zip(axes, ["same_stain", "cross_stain"], ["same stain", "cross-stain"]):
    s = r[r.protocol == prot].set_index("model").reindex(ORDER)
    for col, k, c in zip(["top10", "top5", "top1"], [10, 5, 1], SEQ[::-1]):
        ax.scatter(s[col], y, s=14, color=c, label=f"top-{k}", zorder=3)
    ax.hlines(y, s.top1, s.top10, color="#cfcfcf", lw=1, zorder=2)
    ax.set_title(title, fontsize=8, color=INK); ax.set_xlim(0.25, 1.02)
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
axes[0].set_yticks(y); axes[0].set_yticklabels([NAME[m] for m in ORDER]); axes[0].invert_yaxis()
fig.supxlabel("phone→scanner retrieval accuracy", fontsize=8)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h[::-1], l[::-1], loc="upper center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.6, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(OUT + "fig_retrieval.pdf"); fig.savefig(OUT + "fig_retrieval.png", dpi=200); plt.close(fig)

# 4. kNN-probe phone BA under adaptations (complements Table 3, which reports the linear probe)
c = pd.read_csv(R + "classification.csv")
c = c[(c.classifier == "knn") & (c.eval_set == "phone_all") & (c.metric == "balanced_accuracy") & (c.kind == "metric")]
V = ["baseline", "center", "coral", "fewshot1", "fewshot5", "fewshot10"]
VL = ["None", "Centring", "CORAL", "1-shot", "5-shot", "10-shot"]
M = c.pivot_table(index="model", columns="variant", values="value").reindex(index=ORDER, columns=V)
fig, ax = plt.subplots(figsize=(3.4, 2.9))
im = ax.imshow(M.values, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#cde2fb", "#104281"]),
               vmin=0.6, vmax=1.0, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M.values[i, j]
        if np.isfinite(v):
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=6.5, color="white" if v > 0.85 else INK)
ax.set_xticks(range(len(V))); ax.set_xticklabels(VL, rotation=30, ha="right")
ax.set_yticks(range(len(ORDER))); ax.set_yticklabels([NAME[m] for m in ORDER])
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0)
save(fig, "fig_fixes")
print(M.round(3))
