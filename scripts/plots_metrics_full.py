#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plots_metrics_full.py — gem5 A3 HPC Project Analysis (tailored to your CSV)
Produces:
 - per-config mean ± std bar plots (annotated)
 - per-benchmark per-config plots
 - speedup vs baseline (auto-picked or user-set)
 - summary_stats.csv

Assumptions:
 - CSV: results/experiments_summary.csv
 - Column mapping: 'experiment' -> config label, 'benchmark', 'IPC','CPI',
   'L1D_miss_rate', 'L2_miss_rate', 'simInsts' exist.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---------- CONFIG (change only if you want a specific baseline) ----------
IN = "results/experiments_summary.csv"
OUTDIR = "analysis/plots_metrics"
# If you want to force a baseline experiment name, set it here (string).
# Otherwise set to None and script auto-picks the config with highest mean IPC.
BASELINE_EXPERIMENT = None

# ---------- setup ----------
os.makedirs(OUTDIR, exist_ok=True)
plt.style.use("seaborn-v0_8-deep")

# ---------- read ----------
df = pd.read_csv(IN)
print(f"Loaded {len(df)} rows from {IN}")

# ---------- filter successful runs ----------
if "status" in df.columns:
    df = df[df["status"].astype(str).str.lower() == "ok"].copy()
    print(f"Kept {len(df)} rows with status == ok")
else:
    print("No 'status' column found: proceeding with full dataframe.")

# ---------- create normalized config column ----------
if "experiment" in df.columns:
    df["config"] = df["experiment"].astype(str)
else:
    # fallback: combine some hardware columns if present
    make_cols = [c for c in ("rob","bp","l1d","l1i","l2","issue_width","commit_width") if c in df.columns]
    if make_cols:
        df["config"] = df[make_cols].astype(str).agg("_".join, axis=1)
    else:
        df["config"] = "default"

# ---------- convert numeric columns safely ----------
to_numeric = ["simInsts", "IPC", "CPI", "L1D_miss_rate", "L2_miss_rate"]
for c in to_numeric:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        print(f"Converted column {c} to numeric (NaN where invalid).")
    else:
        print(f"Column {c} not found and will be skipped in plots.")

# ---------- normalize miss-rate scale to percentages if needed ----------
for miss_col in ("L1D_miss_rate", "L2_miss_rate"):
    if miss_col in df.columns:
        mx = df[miss_col].max(skipna=True)
        if pd.notna(mx) and mx <= 1.0:
            df[miss_col] = df[miss_col] * 100.0
            print(f"Interpreted {miss_col} as fraction; converted to percent (x100).")

# ---------- baseline selection ----------
if BASELINE_EXPERIMENT is None:
    if "IPC" in df.columns:
        baseline = df.groupby("config")["IPC"].mean().idxmax()
        print(f"Auto-selected baseline config = '{baseline}' (highest average IPC).")
    else:
        baseline = df["config"].unique()[0]
        print(f"No IPC column found; defaulting baseline to first config: '{baseline}'")
else:
    baseline = BASELINE_EXPERIMENT
    print(f"Using provided baseline: '{baseline}'")

# compute baseline IPC per benchmark (mean if multiple runs)
baseline_ipc = {}
if "IPC" in df.columns:
    baseline_df = df[df["config"] == baseline]
    if not baseline_df.empty:
        baseline_ipc = baseline_df.groupby("benchmark")["IPC"].mean().to_dict()
        print(f"Built baseline IPC values for {len(baseline_ipc)} benchmarks from '{baseline}'.")
    else:
        print(f"Warning: baseline '{baseline}' has no rows in dataframe; speedup will be NaN.")

# ---------- compute speedup column (per-row) ----------
if baseline_ipc:
    def _speedup(row):
        ipc = row.get("IPC", np.nan)
        if pd.isna(ipc):
            return np.nan
        base = baseline_ipc.get(row.get("benchmark"))
        if base is None or pd.isna(base) or base == 0:
            return np.nan
        return ipc / base
    df["Speedup_vs_base"] = df.apply(_speedup, axis=1)
    print("Computed Speedup_vs_base (per-row).")
else:
    df["Speedup_vs_base"] = np.nan

# ---------- plotting helpers ----------
def _annotate_bars(ax, bars, fmt_mean="{:.3f}", extra_text=None):
    for i, b in enumerate(bars):
        h = b.get_height()
        if np.isnan(h):
            continue
        x = b.get_x() + b.get_width() / 2
        txt = fmt_mean.format(h)
        if extra_text:
            txt = txt + "\n" + extra_text[i]
        ax.annotate(txt, (x, h), ha="center", va="bottom", fontsize=8, rotation=0)

def plot_bar_with_error(metric, ylabel, title, per_benchmark=True):
    if metric not in df.columns:
        print(f"Skipping metric '{metric}': not found.")
        return

    # aggregate per config
    agg = df.groupby("config")[metric].agg(["mean","std","count"]).dropna(subset=["mean"])
    if agg.empty:
        print(f"No valid data to plot for '{metric}'.")
        return
    agg = agg.sort_values("mean")
    x = np.arange(len(agg))
    means = agg["mean"].values
    stds = agg["std"].fillna(0).values
    counts = agg["count"].astype(int).values

    cmap = plt.get_cmap("tab20")
    colors = cmap(np.linspace(0, 1, len(agg)))

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(x, means, yerr=stds, capsize=6, color=colors, edgecolor="black", alpha=0.95)
    ax.set_xticks(x)
    ax.set_xticklabels(agg.index, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    # annotations show mean, ±std and n
    extra = [f"±{s:.3f}\nn={n}" for s, n in zip(stds, counts)]
    _annotate_bars(ax, bars, fmt_mean="{:.3f}", extra_text=extra)

    # legend (explanatory)
    legend_handle = Patch(facecolor="lightgray", edgecolor="black")
    ax.legend([legend_handle], [f"bars = mean; errorbar = std; n = runs"], loc="upper right", fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(OUTDIR, f"{metric}_overall.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved overall plot: {out_path}")

    # per-benchmark breakdown
    if per_benchmark:
        for bench, group in df.groupby("benchmark"):
            agg2 = group.groupby("config")[metric].agg(["mean","std","count"]).dropna(subset=["mean"])
            if agg2.empty:
                continue
            agg2 = agg2.sort_values("mean")
            x2 = np.arange(len(agg2))
            means2 = agg2["mean"].values
            stds2 = agg2["std"].fillna(0).values
            counts2 = agg2["count"].astype(int).values
            colors2 = cmap(np.linspace(0, 1, len(agg2)))
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            bars2 = ax2.bar(x2, means2, yerr=stds2, capsize=5, color=colors2, edgecolor="black", alpha=0.95)
            ax2.set_xticks(x2)
            ax2.set_xticklabels(agg2.index, rotation=45, ha="right", fontsize=8)
            ax2.set_ylabel(ylabel)
            ax2.set_title(f"{title} — {bench}")
            ax2.grid(axis="y", linestyle="--", alpha=0.6)
            extra2 = [f"±{s:.3f}\nn={n}" for s, n in zip(stds2, counts2)]
            _annotate_bars(ax2, bars2, fmt_mean="{:.3f}", extra_text=extra2)
            ax2.legend([Patch(facecolor="lightgray", edgecolor="black")], [f"mean ± std"], loc="upper right", fontsize=8)
            plt.tight_layout()
            out2 = os.path.join(OUTDIR, f"{metric}_{bench}.png")
            plt.savefig(out2, dpi=300)
            plt.close()
            # keep it quiet for many benchmarks
    return

# ---------- metrics to produce ----------
metrics = [
    ("IPC", "Instructions per Cycle", "Average IPC by config (mean ± std)"),
    ("CPI", "Cycles per Instruction", "Average CPI by config (mean ± std)"),
    ("L1D_miss_rate", "L1D miss rate (%)", "L1 Data miss rate by config (percent)"),
    ("L2_miss_rate", "L2 miss rate (%)", "L2 miss rate by config (percent)"),
    ("simInsts", "Simulated instructions", "Simulated instruction count by config"),
    ("Speedup_vs_base", "Speedup (relative to baseline)", f"Speedup vs baseline='{baseline}'"),
]

for metric, ylabel, title in metrics:
    plot_bar_with_error(metric, ylabel, title, per_benchmark=True)

# ---------- write summary CSV ----------
summary_cols = [c for c in ["IPC","CPI","L1D_miss_rate","L2_miss_rate","simInsts"] if c in df.columns]
if summary_cols:
    summary = df.groupby("config")[summary_cols].agg(["mean","std","count"]).sort_values(("IPC","mean") if "IPC" in summary_cols else summary_cols[0])
    summary.to_csv(os.path.join(OUTDIR, "summary_stats.csv"))
    print(f"Wrote summary_stats.csv to {OUTDIR}")
else:
    print("No numeric metric columns found to write summary.")

print("All done. Plots + summary (if any) are in:", OUTDIR)
