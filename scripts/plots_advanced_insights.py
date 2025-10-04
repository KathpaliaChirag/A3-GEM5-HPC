#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plots_advanced_insights.py
==========================

Comprehensive HPC/Gem5 A3 analysis suite for IITD project.

Generates:
- Derived metrics (MPKI, HitRate, Speedup, etc.)
- Branch predictor comparison plots
- Scaling trends (ROB, IssueWidth)
- Speedup heatmaps
- Correlation matrix, scatter plots
- Memory-performance tradeoff
- Parallel coordinate overview

Author: CK + GPT-5
"""

import os, sys, re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
from textwrap import dedent

# ---------- CONFIG ----------
IN = "results/experiments_summary.csv"
OUTDIR = "analysis/plots_advanced"
os.makedirs(OUTDIR, exist_ok=True)
DIAG = os.path.join(OUTDIR, "diagnostics.txt")

sns.set_theme(style="whitegrid", context="talk", palette="tab10")
plt.rcParams.update({"figure.autolayout": True})

# ---------- HELPERS ----------
def log(msg):
    with open(DIAG, "a") as f:
        f.write(msg + "\n")
    print(msg)

def save(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    log(f"Saved: {path}")

def enough_var(s, n=2): 
    return s.dropna().nunique() >= n

# ---------- LOAD ----------
if not os.path.exists(IN):
    sys.exit(f"❌ Input file not found: {IN}")
df = pd.read_csv(IN)
if "status" in df.columns:
    df = df[df["status"].astype(str).str.lower() == "ok"].copy()
log(f"✅ Loaded {len(df)} successful runs")

# ---------- CLEAN ----------
df.columns = [c.strip() for c in df.columns]
for c in ["IPC","CPI","L1D_miss_rate","L2_miss_rate"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].max(skipna=True) <= 1:
            df[c] *= 100

# ---------- AUTO PARSE EXPERIMENTS ----------
if "experiment" in df.columns:
    df["experiment"] = df["experiment"].astype(str)
    df["prefetcher"] = df["experiment"].str.extract(r"prefetcher([A-Za-z0-9_]+)", expand=False).fillna("none")
    df["replacement"] = df["experiment"].str.extract(r"(LRU|FIFO|Random|PLRU|MRU)", expand=False).fillna("default")
    for lvl in ["l1d","l2"]:
        if lvl not in df or (df[lvl].isna() | (df[lvl]=="unknown")).all():
            df[lvl] = df["experiment"].str.extract(rf"{lvl}[_-]?([0-9A-Za-z]+)", expand=False).fillna("unk")
    log("🔍 Parsed experiment strings for cache sizes, prefetcher, replacement")

# ---------- DERIVED METRICS ----------
if "IPC" in df.columns:
    df["Speedup_vs_Min"] = df.groupby("benchmark")["IPC"].transform(lambda x: x / x.min())
if "L1D_miss_rate" in df.columns:
    df["HitRate_L1D"] = 100 - df["L1D_miss_rate"]
    df["MPKI_L1D"] = (df["L1D_miss_rate"]/100)*1000/df["IPC"]
if "L2_miss_rate" in df.columns:
    df["HitRate_L2"] = 100 - df["L2_miss_rate"]

# ---------- BRANCH PREDICTOR COMPARISONS ----------
if "bp" in df.columns and enough_var(df["bp"]):
    for bench, g in df.groupby("benchmark"):
        fig, ax = plt.subplots(figsize=(8,5))
        sns.lineplot(data=g, x="rob", y="IPC", hue="bp", marker="o", estimator="mean", errorbar=None, ax=ax)
        ax.set_title(f"ROB scaling — Branch Predictors ({bench})")
        ax.set_xlabel("ROB Size"); ax.set_ylabel("IPC")
        save(fig, f"bp_rob_ipc_{bench}.png")

    fig, ax = plt.subplots(figsize=(8,5))
    sns.lineplot(data=df, x="issue_width", y="IPC", hue="bp", style="benchmark", marker="o", estimator="mean", errorbar=None, ax=ax)
    ax.set_title("Issue-Width Sensitivity of Branch Predictors")
    ax.set_xlabel("Issue Width"); ax.set_ylabel("IPC")
    save(fig, "bp_issuewidth_ipc.png")

    fig = sns.catplot(data=df, x="prefetcher", y="IPC", hue="bp", col="benchmark", kind="bar", height=4, aspect=1)
    plt.subplots_adjust(top=0.85)
    fig.fig.suptitle("Prefetcher × Branch Predictor Interaction (IPC)")
    fig.savefig(os.path.join(OUTDIR, "bp_prefetcher_interaction.png"), dpi=300)
    plt.close(fig.fig)
    log("✅ Branch predictor comparison plots complete")

# ---------- SPEEDUP HEATMAP ----------
if "rob" in df.columns and "bp" in df.columns:
    base = df[df["bp"].str.lower()=="none"]
    if not base.empty:
        merged = df.merge(base[["benchmark","rob","issue_width","IPC"]], on=["benchmark","rob","issue_width"], suffixes=("","_none"))
        merged["Speedup_vs_NoneBP"] = merged["IPC"]/merged["IPC_none"]
        pivot = merged.pivot_table(values="Speedup_vs_NoneBP", index="rob", columns="bp", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(8,5))
        sns.heatmap(pivot, annot=True, cmap="YlGnBu", ax=ax)
        ax.set_title("Speedup vs No Predictor Baseline (avg over benchmarks)")
        save(fig, "bp_speedup_heatmap.png")

# ---------- CORRELATION HEATMAP ----------
corr_cols = [c for c in ["IPC","CPI","L1D_miss_rate","L2_miss_rate","rob","issue_width","commit_width"] if c in df.columns]
if len(corr_cols) > 3:
    corr = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(7,6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Parameter Correlation Heatmap")
    save(fig, "corr_heatmap.png")

# ---------- MEMORY–PERFORMANCE TRADEOFF ----------
if "L1D_miss_rate" in df.columns and "IPC" in df.columns:
    fig, ax = plt.subplots(figsize=(7,5))
    sns.scatterplot(data=df, x="L1D_miss_rate", y="IPC", hue="bp", style="benchmark", ax=ax)
    ax.set_title("IPC vs L1D Miss Rate (speculation–memory correlation)")
    ax.set_xlabel("L1D Miss Rate (%)"); ax.set_ylabel("IPC")
    save(fig, "ipc_vs_l1d_missrate.png")

# ---------- SPEEDUP SCALING PLOTS ----------
if "rob" in df.columns:
    fig, ax = plt.subplots(figsize=(8,5))
    sns.lineplot(data=df, x="rob", y="Speedup_vs_Min", hue="benchmark", marker="o", estimator="mean", errorbar=None, ax=ax)
    ax.set_title("Speedup vs ROB Size (normalized to min ROB per benchmark)")
    ax.set_xlabel("ROB Size"); ax.set_ylabel("Relative Speedup")
    save(fig, "speedup_vs_rob.png")

if "issue_width" in df.columns:
    fig, ax = plt.subplots(figsize=(8,5))
    sns.lineplot(data=df, x="issue_width", y="Speedup_vs_Min", hue="benchmark", marker="o", estimator="mean", errorbar=None, ax=ax)
    ax.set_title("Speedup vs Issue Width (normalized to min)")
    ax.set_xlabel("Issue Width"); ax.set_ylabel("Relative Speedup")
    save(fig, "speedup_vs_issuewidth.png")

# ---------- PARALLEL COORDINATES (DESIGN SPACE) ----------
sel_cols = [c for c in ["rob","issue_width","commit_width","IPC"] if c in df.columns]
if len(sel_cols) == 4:
    try:
        tmp = df[sel_cols].dropna().copy()
        # Convert ROB to categorical for plotting
        tmp["rob"] = tmp["rob"].astype(str)
        tmp["issue_width"] = pd.to_numeric(tmp["issue_width"], errors="coerce")
        tmp["commit_width"] = pd.to_numeric(tmp["commit_width"], errors="coerce")
        tmp["IPC"] = pd.to_numeric(tmp["IPC"], errors="coerce")

        fig = plt.figure(figsize=(9,6))
        parallel_coordinates(tmp, class_column="rob", colormap=plt.cm.viridis)
        plt.title("Design Space Overview (ROB–Issue–Commit–IPC)")
        plt.ylabel("IPC")
        plt.xlabel("Configuration Set (colored by ROB size)")
        save(fig, "parallel_coords_designspace.png")
    except Exception as e:
        log(f"⚠️ Skipped parallel coordinates due to: {e}")

# ---------- DIAGNOSTICS ----------
summary = dedent(f"""
✅ Advanced analysis complete.
Plots directory: {OUTDIR}
Derived metrics: MPKI_L1D, HitRate_L1D/L2, Speedup_vs_Min, Speedup_vs_NoneBP
Branch predictors compared: {df['bp'].nunique() if 'bp' in df.columns else 0}
Prefetchers: {df['prefetcher'].nunique() if 'prefetcher' in df.columns else 0}
Benchmarks: {df['benchmark'].nunique() if 'benchmark' in df.columns else 0}

Open PNGs in {OUTDIR}/ for detailed plots.
""").strip()
log(summary)
open(os.path.join(OUTDIR,"report.txt"),"w").write(summary)
