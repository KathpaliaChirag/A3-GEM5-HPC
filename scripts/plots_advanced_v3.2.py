#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plots_advanced_v3.2.py
Final stable version — fixes seaborn scatter legend crash & adds robustness for legend creation.
"""

import os, sys, math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------- CONFIG ----------------
INPUT = "results/experiments_summary.csv"
OUTDIR = "analysis/plots_v3_2"
os.makedirs(OUTDIR, exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=1.4)
plt.rcParams.update({
    "figure.figsize": (14, 9),
    "savefig.dpi": 300,
    "axes.titlesize": 20,
    "axes.labelsize": 15,
    "legend.fontsize": 13
})
PALETTE = "tab10"

def log(msg):
    print(msg)
    with open(os.path.join(OUTDIR, "diagnostics.txt"), "a") as f:
        f.write(msg + "\n")

def save(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved: {path}")

# ---------------- LOAD ----------------
if not os.path.exists(INPUT):
    sys.exit(f"❌ Missing {INPUT}")
df = pd.read_csv(INPUT)
df = df[df.get("status","ok") == "ok"].copy()
log(f"✅ Loaded {len(df)} ok runs")

df.columns = [c.strip() for c in df.columns]

# numeric conversions
for c in ["IPC","CPI","L1D_miss_rate","L2_miss_rate","rob","issue_width","commit_width"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# convert miss rates to %
for m in ["L1D_miss_rate","L2_miss_rate"]:
    if m in df and df[m].max(skipna=True) <= 1:
        df[m] *= 100

# derived metrics
if "L1D_miss_rate" in df: df["HitRate_L1D"] = 100 - df["L1D_miss_rate"]
if "L2_miss_rate" in df: df["HitRate_L2"] = 100 - df["L2_miss_rate"]

# ---------------- UTILITIES ----------------
def rob_to_size(series):
    """Convert ROB size column to numeric bubble size array."""
    try:
        s = pd.to_numeric(series, errors="coerce").fillna(0)
        rmin, rmax = np.nanmin(s), np.nanmax(s)
        if rmax == rmin:
            return np.repeat(120.0, len(s))
        norm = (s - rmin) / (rmax - rmin)
        return (80.0 + norm * 200.0).astype(float)
    except Exception:
        return np.repeat(100.0, len(series))

def add_custom_legend(ax, rob_vals):
    """Custom bubble legend for ROB sizes."""
    handles, labels = [], []
    rob_vals = sorted([v for v in rob_vals if not pd.isna(v)])
    for r in rob_vals[:3]:
        handles.append(Line2D([], [], marker='o', color='w', markerfacecolor='gray',
                              markersize=math.sqrt(rob_to_size(pd.Series([r]))[0]),
                              markeredgecolor='k', linestyle=''))
        labels.append(f"{int(r)}")
    ax.legend(handles, labels, title="ROB size", bbox_to_anchor=(1.02, 0.5), loc="center left")

# ---------------- PLOTS ----------------
#def plot_scatter_ipc_vs_l1d():
   # if not {"IPC","L1D_miss_rate"}.issubset(df.columns): return
  #  # Convert size array to flat numpy float array
 #   sizes = rob_to_size(df.get("rob", pd.Series([np.nan]*len(df)))).astype(float)
#    fig, ax = plt.subplots(figsize=(14,9))
    #sns.scatterplot(
   #     data=df,
  #      x="L1D_miss_rate",
 #       y="IPC",
#        hue="benchmark" if "benchmark" in df else None,
        #style="benchmark" if "benchmark" in df else None,
       # s=sizes.tolist(),  # Ensure it’s a pure list of floats
      #  alpha=0.85,
     #   palette=PALETTE,
    #    legend="brief",
   #     ax=ax
  #  )
 #   ax.set_title("IPC vs L1D Miss Rate — Memory Locality Impact (Bubble size ≈ ROB)")
#    ax.set_xlabel("L1D Miss Rate (%)")
    #ax.set_ylabel("IPC")
   # if "rob" in df and df["rob"].notna().any():
  #      add_custom_legend(ax, df["rob"].dropna().unique())
 #   ax.legend(bbox_to_anchor=(1.02,1), loc="upper left")
#    save(fig, "ipc_vs_l1dmiss_scatter_fixed.png")


def plot_scatter_ipc_vs_l1d(df):
    if not {"L1D_miss_rate", "IPC", "rob"}.issubset(df.columns):
        print("⚠️ Required columns not found for IPC vs L1D plot")
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    df = df.copy()
    df["rob_scaled"] = (
        (df["rob"] - df["rob"].min()) / (df["rob"].max() - df["rob"].min()) * 250 + 50
    )
    sns.scatterplot(
        data=df,
        x="L1D_miss_rate",
        y="IPC",
        hue="benchmark",
        size="rob_scaled",
        sizes=(50, 300),
        alpha=0.85,
        ax=ax,
        legend="brief"
    )
    ax.set_title("IPC vs L1D Miss Rate (Bubble Size ∝ ROB)")
    ax.set_xlabel("L1D Miss Rate (%)")
    ax.set_ylabel("IPC")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    plt.show()

def plot_issuewidth_ipc():
    if not {"issue_width","IPC"}.issubset(df.columns): return
    fig, ax = plt.subplots(figsize=(13,8))
    sns.lineplot(
        data=df, x="issue_width", y="IPC",
        hue="commit_width" if "commit_width" in df else None,
        marker="o", palette=PALETTE, ax=ax, errorbar=None
    )
    for line in ax.get_lines():
        for x, y in zip(line.get_xdata(), line.get_ydata()):
            if pd.notna(y):
                ax.annotate(f"{y:.3f}", xy=(x,y), xytext=(0,6),
                            textcoords="offset points", ha="center", fontsize=11)
    ax.set_title("Issue Width vs IPC (Commit Width colored)")
    ax.set_xlabel("Issue Width"); ax.set_ylabel("IPC")
    ax.legend(title="Commit Width", bbox_to_anchor=(1.02,1), loc="upper left")
    save(fig, "issuewidth_vs_ipc_fixed.png")

def plot_memdep_ipc():
    if "memdep" not in df or "IPC" not in df: return
    fig, ax = plt.subplots(figsize=(12,8))
    sns.barplot(
        data=df, x="memdep", y="IPC",
        hue="benchmark" if "benchmark" in df else None,
        palette=PALETTE, estimator=np.mean, errorbar=None, ax=ax
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=4)
    ax.set_title("Memory Dependence Predictor (On/Off) vs IPC")
    ax.set_xlabel("MemDep"); ax.set_ylabel("IPC")
    ax.legend(bbox_to_anchor=(1.02,1), loc="upper left")
    save(fig, "memdep_vs_ipc_fixed.png")

def plot_assoc_vs_hit():
    if not {"assoc","HitRate_L1D"}.issubset(df.columns): return
    fig, ax = plt.subplots(figsize=(13,8))
    sns.lineplot(
        data=df, x="assoc", y="HitRate_L1D",
        hue="benchmark" if "benchmark" in df else None,
        marker="o", palette=PALETTE, ax=ax, errorbar=None
    )
    ax.set_title("Associativity vs L1D Hit Rate")
    ax.set_xlabel("Associativity"); ax.set_ylabel("L1D Hit Rate (%)")
    ax.legend(bbox_to_anchor=(1.02,1), loc="upper left")
    save(fig, "assoc_vs_l1d_hitrate_fixed.png")

# ---------------- RUN ALL ----------------
plot_scatter_ipc_vs_l1d(df)
plot_issuewidth_ipc()
plot_memdep_ipc()
plot_assoc_vs_hit()

log("✅ v3.2 plots generated successfully (scatter bug fixed).")
