#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plots_metrics_full.py — gem5 A3 HPC Project Analysis

Generates detailed performance plots from experiments_summary.csv:
  ✅ IPC & CPI
  ✅ Cache miss rates (L1, L2)
  ✅ Simulation time
  ✅ Speedup vs baseline
  ✅ Benchmark-wise comparisons

Author: CK + GPT-5
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------- Config ----------
IN = "results/experiments_summary.csv"
OUTDIR = "analysis/plots_metrics"
BASELINE_CONFIG = "config_base"  # ⚙️ change this to your baseline config name

os.makedirs(OUTDIR, exist_ok=True)
plt.style.use("seaborn-v0_8-deep")  # modern color palette

# ---------- Read & Clean Data ----------
df = pd.read_csv(IN)
print(f"📘 Loaded {len(df)} rows from {IN}")

df = df[df["status"] == "ok"].copy()
print(f"✅ Retained {len(df)} successful runs")

# ---------- Derived Metrics ----------
if "sim_insts" in df.columns and "sim_seconds" in df.columns:
    df["IPC"] = df["sim_insts"] / df["sim_seconds"]
    df["CPI"] = 1 / df["IPC"]

# Compute Speedup vs baseline (if applicable)
if BASELINE_CONFIG in df["config"].unique():
    baseline_ipc = (
        df[df["config"] == BASELINE_CONFIG]
        .set_index("benchmark")["IPC"]
        .to_dict()
    )
    df["Speedup_vs_base"] = df.apply(
        lambda r: r["IPC"] / baseline_ipc.get(r["benchmark"], np.nan), axis=1
    )

# ---------- Utility: Pretty Plot Function ----------
def annotate_bars(ax, fmt="{:.2f}"):
    """Add value labels above bars"""
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height):
            ax.annotate(fmt.format(height),
                        (p.get_x() + p.get_width() / 2, height),
                        ha="center", va="bottom", fontsize=9, color="black")

def plot_metric(metric, ylabel, title, ylim=None, per_benchmark=False):
    if metric not in df.columns:
        print(f"⚠️ Skipping missing metric: {metric}")
        return

    # --- Per config (mean over benchmarks) ---
    plt.figure(figsize=(12, 6))
    grouped = df.groupby("config")[metric].mean().sort_values()
    ax = grouped.plot(kind="bar", color=plt.cm.tab20(np.linspace(0, 1, len(grouped))),
                      edgecolor="black", alpha=0.9)
    annotate_bars(ax)
    plt.ylabel(ylabel)
    plt.title(title)
    if ylim: plt.ylim(ylim)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(["Average over benchmarks"], loc="upper right")
    plt.tight_layout()
    outfile = os.path.join(OUTDIR, f"{metric}_overall.png")
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"📊 Saved: {outfile}")

    # --- Per benchmark (optional) ---
    if per_benchmark:
        for bench, group in df.groupby("benchmark"):
            plt.figure(figsize=(10, 5))
            sub = group.groupby("config")[metric].mean().sort_values()
            ax = sub.plot(kind="bar", color=plt.cm.Paired(np.linspace(0, 1, len(sub))),
                          edgecolor="black", alpha=0.9)
            annotate_bars(ax)
            plt.ylabel(ylabel)
            plt.title(f"{title} — {bench}")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.legend(["Per config"], loc="upper right")
            plt.tight_layout()
            outfile = os.path.join(OUTDIR, f"{metric}_{bench}.png")
            plt.savefig(outfile, dpi=300)
            plt.close()

# ---------- Metrics to Plot ----------
metrics_to_plot = [
    ("IPC", "Instructions per Cycle", "Average IPC by Config"),
    ("CPI", "Cycles per Instruction", "Average CPI by Config"),
    ("system.cpu.l1.overall_miss_rate::total", "L1 Miss Rate", "L1 Miss Rate by Config"),
    ("system.l2.overall_miss_rate::total", "L2 Miss Rate", "L2 Miss Rate by Config"),
    ("sim_seconds", "Simulation Time (s)", "Execution Time by Config"),
    ("Speedup_vs_base", "Speedup (vs baseline)", "Relative Speedup by Config"),
]

# ---------- Generate Plots ----------
for metric, ylabel, title in metrics_to_plot:
    plot_metric(metric, ylabel, title, per_benchmark=True)

# ---------- Summary Report ----------
summary = (
    df.groupby("config")[["IPC", "CPI", "sim_seconds"]]
    .mean()
    .sort_values("IPC", ascending=False)
)
summary.to_csv(os.path.join(OUTDIR, "summary_stats.csv"))
print("📈 Summary stats saved to summary_stats.csv")

print("\n🎯 Done! All plots and summary written to:", OUTDIR)
