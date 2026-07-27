#!/usr/bin/env python3
"""
Combined Spearman rho heatmap for summeval / fluency across all frontier judges.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import seaborn as sns

BASE = Path(__file__).resolve().parent
RANGES = ["[0,1]", "[1,3]", "[1,5]", "[0,10]", "[0,100]"]
JUDGES = [
    ("gemini31_pro", "Gemini 3.1 Pro"),
    ("gpt54", "GPT 5.4"),
    ("opus47", "Claude Opus 4.7"),]

CMAP   = "RdYlGn"
VMIN   = 0.0
VMAX   = 1.0
OUT_PDF = BASE / "summeval_flu_spearman_combined.pdf"


def build_matrix(row: pd.Series) -> np.ndarray:
    n = len(RANGES)
    mat = np.eye(n, dtype=float)
    for i, ri in enumerate(RANGES):
        for j, rj in enumerate(RANGES):
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            col = f"spearman_{RANGES[a]}__{RANGES[b]}"
            if col in row.index:
                mat[i, j] = float(row[col])
    return mat

df = pd.read_csv(BASE.parent / "reports" / "cross_range_correlation_summary.csv")
flu_df = df[(df["dataset"] == "summeval") & (df["metric"] == "flu")].set_index("model")

FONT = 14          
TITLE_FONT = 15    
CBAR_FONT  = 15    

n_judges = len(JUDGES)

fig = plt.figure(figsize=(5.5 * n_judges + 1.6, 5.4), dpi=220)
gs = gridspec.GridSpec(
    1, n_judges + 1,
    width_ratios=[5.5] * n_judges + [0.35],
    wspace=0.08,
)

axes = [fig.add_subplot(gs[0, k]) for k in range(n_judges)]
cbar_ax = fig.add_subplot(gs[0, n_judges])

norm = Normalize(vmin=VMIN, vmax=VMAX)
cmap_obj = plt.get_cmap(CMAP)

for k, (model_dir, model_name) in enumerate(JUDGES):
    ax = axes[k]

    if model_dir not in flu_df.index:
        ax.set_visible(False)
        continue

    mat = build_matrix(flu_df.loc[model_dir])
    panel_df = pd.DataFrame(mat, index=RANGES, columns=RANGES)

    annot = panel_df.copy().astype(object)
    for i in range(panel_df.shape[0]):
        for j in range(panel_df.shape[1]):
            v = panel_df.iat[i, j]
            annot.iat[i, j] = "" if np.isnan(v) else f"{v:.2f}"

    sns.heatmap(
        panel_df,
        annot=annot,
        fmt="",
        cmap=CMAP,
        vmin=VMIN,
        vmax=VMAX,
        ax=ax,
        linewidths=0.4,
        linecolor="#d8d8d8",
        annot_kws={"size": FONT},
        cbar=False,          
    )

    ax.set_title(model_name, fontsize=TITLE_FONT, fontweight="bold", pad=10)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=FONT)

    if k == 0:
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=FONT)
        ax.set_ylabel("Score range", fontsize=FONT)
    else:
        ax.set_yticklabels([])
        ax.set_ylabel("")

    ax.set_xlabel("Score range", fontsize=FONT)

sm = ScalarMappable(cmap=cmap_obj, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("Spearman rho", fontsize=CBAR_FONT, labelpad=10)
cbar.ax.tick_params(labelsize=FONT - 1)

fig.savefig(OUT_PDF, bbox_inches="tight")
plt.close(fig)
