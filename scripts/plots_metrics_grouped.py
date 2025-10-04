#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plots_metrics_grouped.py — Clean grouped plots for gem5 HPC A3 project

This script:
✅ Reads experiments_summary.csv
✅ Aggregates metrics (mean ± std) by key parameters (ROB, BP, L1D, L2, etc.)
✅ Plots clean, grouped bar charts (IPC, CPI, miss rates, etc.)
✅ Annotates bars with mean/std values
✅ Saves all results in analysis/plots_grouped/

Author: CK + GPT-5
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---------- Config ----------
IN = "results/experiments_summary.csv"
OUTDIR = "analysis/plots_grouped"
os.makedirs(OUTDIR, exist_ok=True)

plt.style.use("ggplot")
plt.rcParams.update({
    "figure.autolayout": True,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "font.family": "DejaVu Sans",
})

# ---------- Load Data ----------
df = pd.read_csv(IN)
df = df[df["status"].astype(str).str.lower() == "ok"].copy()
print(f"✅ Loaded {len(df)} successful runs from {IN}")

# ---------- Clean numeric columns ----------
numeric_cols = ["IPC", "CPI", "L1D_miss_rate", "L2_miss_rate", "simInsts"]
for c in numeric_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Convert miss rates to % if they’re 0–1
for miss_col in ["L1D_miss_rate", "L2_miss_rate"]:
    if miss_col in df.columns and df[miss_col].max(skipna=True) <= 1.0:
        df[miss_col] *= 100

# ---------- Define metrics and parameters ----------
metrics = [
    ("IPC", "Instructions per Cycle", "Average IPC"),
    ("CPI", "Cycles per Instruction", "Average CPI"),
    ("L1D_miss_rate", "L1D Miss Rate (%)", "Average L1D Miss Rate"),
    ("L2_miss_rate", "L2 Miss Rate (%)", "Average L2 Miss Rate"),
]

# We'll aggregate results by these key parameters:
group_params = ["rob", "bp", "l1d", "l1i", "l2", "issue_width", "commit_width"]

# ---------- Plot helper ----------
def annotate_bars(ax, bars, stds):
    """Add mean ± std labels above bars"""
    for bar, s in zip(bars, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height,
                f"{height:.3f}\n±{s:.3f}",
                ha="center", va="bottom", fontsize=8, color="black")

def plot_grouped(metric, ylabel, title):
    """Plot metric aggregated by each architecture parameter"""
    if metric not in df.columns:
        print(f"⚠️ Metric {metric} not found, skipping.")
        return

    for param in group_params:
        if param not in df.columns:
            continue

        agg = df.groupby(param)[metric].agg(["mean", "std", "count"]).dropna()
        if len(agg) < 2:
            continue  # skip if only one unique value
        agg = agg.sort_index()

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(agg))
        colors = plt.cm.tab20(np.linspace(0, 1, len(agg)))

        bars = ax.bar(x, agg["mean"], yerr=agg["std"], color=colors,
                      edgecolor="black", alpha=0.9, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(agg.index.astype(str), rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} grouped by {param}")
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        annotate_bars(ax, bars, agg["std"].fillna(0).values)
        ax.legend([Patch(facecolor="lightgray", edgecolor="black")],
                  ["mean ± std"], loc="upper right", fontsize=8)
        plt.tight_layout()
        out = os.path.join(OUTDIR, f"{metric}_by_{param}.png")
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"📊 Saved plot: {out}")

# ---------- Generate plots ----------
for metric, ylabel, title in metrics:
    plot_grouped(metric, ylabel, title)

# ---------- Write summary table ----------
summary = df.groupby(["rob", "bp", "l1d", "l2"]).agg({
    "IPC": "mean", "CPI": "mean", "L1D_miss_rate": "mean", "L2_miss_rate": "mean"
}).reset_index()
summary.to_csv(os.path.join(OUTDIR, "aggregated_summary.csv"), index=False)
print("✅ Saved aggregated summary to analysis/plots_grouped/aggregated_summary.csv")

print("\n🎉 All clean grouped plots generated successfully!")
