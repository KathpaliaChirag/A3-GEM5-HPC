#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plots_final_report.py
=====================
Final explanatory plotting suite for gem5 HPC Assignment-3.
- Auto-extracts hidden parameters (prefetcher, replacement, cache sizes)
- Skips invalid/empty plots automatically
- Saves valid, explanatory graphs and summaries

Outputs:
  analysis/plots_final/
    *.png                → explanatory graphs
    diagnostics.txt      → reasons for skipped plots
    9_best_config*.csv   → best IPC configs
    report.txt           → summary
"""

import os, sys, re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from textwrap import dedent

# ---------- CONFIG ----------
IN = "results/experiments_summary.csv"
OUTDIR = "analysis/plots_final"
os.makedirs(OUTDIR, exist_ok=True)
DIAG = os.path.join(OUTDIR, "diagnostics.txt")

sns.set_theme(style="whitegrid", context="talk", palette="tab10")
plt.rcParams.update({"figure.autolayout": True})

# ---------- BASIC HELPERS ----------
def write_diag(msg):
    with open(DIAG, "a") as f:
        f.write(msg + "\n")
    print(msg)

def save_fig(fig, name):
    out = os.path.join(OUTDIR, name)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    write_diag(f"Saved plot: {out}")

def enough_var(series, min_unique=2):
    """True if series has >= min_unique unique non-null values"""
    return series.dropna().nunique() >= min_unique

# ---------- READ & CLEAN ----------
if not os.path.exists(IN):
    sys.exit(f"❌ ERROR: {IN} not found")

df = pd.read_csv(IN)
orig_len = len(df)
if "status" in df.columns:
    df = df[df["status"].astype(str).str.lower() == "ok"].copy()
good_len = len(df)

# Strip whitespace in headers
df.columns = [c.strip() for c in df.columns]

# Convert numerics
for c in ["IPC","CPI","simInsts","L1D_miss_rate","L2_miss_rate"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Convert miss-rates to %
for m in ("L1D_miss_rate","L2_miss_rate"):
    if m in df.columns and df[m].max(skipna=True) <= 1:
        df[m] *= 100
        write_diag(f"Converted {m} from fraction to percent (×100).")

# Ensure essential columns exist
key_cols = ["rob","bp","l1d","l1i","l2","issue_width",
            "commit_width","experiment","benchmark"]
for k in key_cols:
    if k not in df.columns:
        df[k] = "unknown"

# ---------- AUTO-PARSE experiment column ----------
if "experiment" in df.columns:
    df["experiment"] = df["experiment"].astype(str)
    # Prefetcher
    df["prefetcher"] = df["experiment"].str.extract(
        r"prefetcher([A-Za-z0-9_]+)", expand=False).fillna("none")
    # Replacement
    df["replacement"] = df["experiment"].str.extract(
        r"(LRU|FIFO|Random|PLRU|MRU)", expand=False).fillna("default")
    # L1/L2 sizes if missing
    if (df["l1d"] == "unknown").all():
        df["l1d"] = df["experiment"].str.extract(r"l1d([0-9A-Za-z]+)", expand=False).fillna("unk")
    if (df["l2"] == "unknown").all():
        df["l2"] = df["experiment"].str.extract(r"l2[_-]?([0-9A-Za-z]+)", expand=False).fillna("unk")
    write_diag("Auto-parsed experiment strings for prefetcher, replacement, cache sizes.")

# ---------- STYLE ----------
def finalize_axis(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if ax.get_legend() is not None:
        ax.legend(loc="best", fontsize=9)
    for lbl in ax.get_xticklabels(): lbl.set_rotation(30)
    return ax

# ---------- PLOT 1: ROB vs IPC per benchmark ----------
def plot_rob_ipc():
    write_diag("\nPlot 1 – ROB vs IPC (colored by Issue Width)")
    if not {"rob","IPC","issue_width","benchmark"} <= set(df.columns):
        write_diag("Skipping Plot 1: missing columns"); return
    for bench, g in df.groupby("benchmark"):
        if not (enough_var(g["rob"]) and enough_var(g["IPC"])): continue
        fig, ax = plt.subplots(figsize=(8,5))
        sns.lineplot(data=g, x="rob", y="IPC", hue="issue_width",
                     marker="o", estimator="mean", errorbar=None, ax=ax)
        finalize_axis(ax, f"ROB scaling — {bench}",
                      xlabel="ROB size", ylabel="IPC")
        save_fig(fig, f"1_rob_ipc_{bench}.png")

# ---------- PLOT 2: Issue width vs IPC/CPI ----------
def plot_issuewidth_ipc_cpi():
    write_diag("\nPlot 2 – Issue Width scaling (IPC + CPI)")
    if "issue_width" not in df.columns: return
    fig, ax1 = plt.subplots(figsize=(8,5))
    sns.lineplot(data=df, x="issue_width", y="IPC", hue="commit_width",
                 marker="o", estimator="mean", errorbar=None, ax=ax1)
    ax2 = ax1.twinx()
    sns.lineplot(data=df, x="issue_width", y="CPI", hue="commit_width",
                 marker="s", linestyle="--", estimator="mean", errorbar=None,
                 legend=False, ax=ax2)
    ax1.set_ylabel("IPC"); ax2.set_ylabel("CPI")
    finalize_axis(ax1, "Issue Width vs IPC / CPI (grouped by Commit Width)",
                  xlabel="Issue Width")
    save_fig(fig, "2_issuewidth_ipc_cpi_commitwidth.png")

# ---------- PLOT 3: L1D vs L1D Miss Rate ----------
def plot_l1d_missrate():
    write_diag("\nPlot 3 – L1D size vs Miss Rate (Prefetcher/BP)")
    if not {"L1D_miss_rate","l1d"} <= set(df.columns):
        write_diag("Skipping 3: missing columns"); return
    if not enough_var(df["l1d"]): write_diag("Skipping 3: single L1D value"); return
    hue = "prefetcher" if enough_var(df["prefetcher"]) else "bp"
    fig, ax = plt.subplots(figsize=(8,5))
    sns.lineplot(data=df, x="l1d", y="L1D_miss_rate", hue=hue,
                 marker="o", estimator="mean", errorbar=None, ax=ax)
    finalize_axis(ax, "L1D size vs Miss Rate (by Prefetcher/BP)",
                  "L1D size", "L1D Miss Rate (%)")
    save_fig(fig, "3_l1d_missrate_prefetcher.png")

# ---------- PLOT 4: L2 vs L2 Miss Rate ----------
def plot_l2_missrate():
    write_diag("\nPlot 4 – L2 size vs Miss Rate (by L1D)")
    if not {"L2_miss_rate","l2"} <= set(df.columns): return
    if not enough_var(df["l2"]): return
    fig, ax = plt.subplots(figsize=(8,5))
    sns.lineplot(data=df, x="l2", y="L2_miss_rate", hue="l1d",
                 marker="o", estimator="mean", errorbar=None, ax=ax)
    finalize_axis(ax, "L2 size vs Miss Rate (by L1D)", "L2 size", "L2 Miss Rate (%)")
    save_fig(fig, "4_l2_missrate_assoc.png")

# ---------- PLOT 5: Prefetcher vs IPC ----------
def plot_prefetcher_ipc():
    write_diag("\nPlot 5 – Prefetcher Type vs IPC (grouped by BP)")
    if "prefetcher" not in df.columns or not enough_var(df["prefetcher"]):
        write_diag("Skipping 5: no prefetcher info"); return
    if not enough_var(df.get("IPC", pd.Series(dtype=float))):
        write_diag("Skipping 5: no IPC data"); return
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(data=df, x="prefetcher", y="IPC", hue="bp",
                estimator="mean", errorbar=None, ax=ax)
    finalize_axis(ax, "Prefetcher Type vs IPC (by BP)", "Prefetcher", "IPC")
    save_fig(fig, "5_prefetcher_ipc_bp.png")

# ---------- PLOT 6: BP vs IPC/CPI ----------
def plot_bp():
    write_diag("\nPlot 6 – Branch Predictor vs IPC/CPI")
    if "bp" not in df.columns or not enough_var(df["bp"]): return
    for metric in ["IPC","CPI"]:
        if metric in df.columns and enough_var(df[metric]):
            fig, ax = plt.subplots(figsize=(8,5))
            sns.barplot(data=df, x="bp", y=metric,
                        estimator="mean", errorbar=None, ax=ax)
            finalize_axis(ax, f"Branch Predictor vs {metric}", "Branch Predictor", metric)
            save_fig(fig, f"6_bp_{metric.lower()}.png")

# ---------- PLOT 7: Replacement vs IPC ----------
def plot_replacement():
    write_diag("\nPlot 7 – Replacement Policy vs IPC (grouped by L2)")
    if "replacement" not in df.columns or not enough_var(df["replacement"]): return
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(data=df, x="replacement", y="IPC", hue="l2",
                estimator="mean", errorbar=None, ax=ax)
    finalize_axis(ax, "Replacement Policy vs IPC (by Cache Size)", "Replacement", "IPC")
    save_fig(fig, "7_replacement_ipc_cache.png")

# ---------- PLOT 8: Failure Heatmap ----------
def plot_failure_heatmap():
    write_diag("\nPlot 8 – Failure Rate Heatmap")
    all_df = pd.read_csv(IN)
    all_df["failed"] = all_df["status"].astype(str).str.lower().apply(lambda s: 0 if s=="ok" else 1)
    if "benchmark" not in all_df.columns or "rob" not in all_df.columns: return
    pivot = all_df.groupby(["benchmark","rob"])["failed"].mean().unstack().fillna(0)
    fig, ax = plt.subplots(figsize=(10, max(4,0.5*pivot.shape[0])))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Reds", ax=ax)
    finalize_axis(ax, "Failure Rate per Benchmark × ROB", "ROB", "Benchmark")
    save_fig(fig, "8_failure_rate_heatmap.png")

# ---------- PLOT 9: Best Config Summary ----------
def best_config():
    write_diag("\nPlot 9 – Best IPC per Benchmark")
    if "IPC" not in df.columns or "benchmark" not in df.columns: return
    idx = df.groupby("benchmark")["IPC"].idxmax()
    best = df.loc[idx, ["benchmark","experiment","IPC","rob","bp","l1d","l2",
                        "issue_width","commit_width"]].sort_values("benchmark")
    out = os.path.join(OUTDIR, "9_best_config_per_benchmark.csv")
    best.to_csv(out, index=False)
    write_diag(f"Saved best-config table: {out}")

# ---------- RUN ----------
open(DIAG,"w").write("Diagnostics\n" + "="*60 + "\n")
write_diag(f"Loaded {orig_len} rows, {good_len} status==ok")

plot_rob_ipc()
plot_issuewidth_ipc_cpi()
plot_l1d_missrate()
plot_l2_missrate()
plot_prefetcher_ipc()
plot_bp()
plot_replacement()
plot_failure_heatmap()
best_config()

# ---------- SUMMARY ----------
report = dedent(f"""
✅ Plots completed.
Total rows: {orig_len}
Valid rows (status == ok): {good_len}
Output dir: {OUTDIR}
Diagnostics: {DIAG}

Inspect diagnostics.txt for skipped plots reasons.
""").strip()

open(os.path.join(OUTDIR, "report.txt"),"w").write(report+"\n")
write_diag("\n" + report)
