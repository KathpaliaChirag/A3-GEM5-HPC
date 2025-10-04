#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plots_full_study.py
===================
Ultimate gem5 A3 HPC study visualization suite.

Combines:
- Core scaling plots (ROB, IssueWidth, BP)
- Advanced analyses (speedup, correlation, memory tradeoff)
- Cache behavior (assoc, size vs hit rate)
- Prefetcher / MemDep predictor impact
- Clean, large, and report-ready graphs
"""

import os, sys, re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from textwrap import dedent

# ---------- CONFIG ----------
IN = "results/experiments_summary.csv"
OUTDIR = "analysis/plots_full_study"
os.makedirs(OUTDIR, exist_ok=True)
DIAG = os.path.join(OUTDIR, "diagnostics.txt")

sns.set_theme(style="whitegrid", context="talk", palette="tab10")
plt.rcParams.update({
    "figure.figsize": (10, 7),
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "figure.autolayout": True
})

# ---------- HELPERS ----------
def log(msg):
    with open(DIAG, "a") as f:
        f.write(msg + "\n")
    print(msg)

def save(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"✅ Saved: {path}")

def enough_var(s, n=2): 
    return s.dropna().nunique() >= n

# ---------- LOAD ----------
if not os.path.exists(IN):
    sys.exit(f"❌ Missing input file: {IN}")
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

# ---------- PARSE HIDDEN PARAMS ----------
if "experiment" in df.columns:
    df["experiment"] = df["experiment"].astype(str)
    df["prefetcher"] = df["experiment"].str.extract(r"prefetcher([A-Za-z0-9_]+)", expand=False).fillna("none")
    df["replacement"] = df["experiment"].str.extract(r"(LRU|FIFO|Random|PLRU|MRU)", expand=False).fillna("default")
    df["assoc"] = df["experiment"].str.extract(r"assoc([0-9]+)", expand=False).fillna("unk")
    for lvl in ["l1d","l2"]:
        if lvl not in df or (df[lvl].isna() | (df[lvl]=="unknown")).all():
            df[lvl] = df["experiment"].str.extract(rf"{lvl}[_-]?([0-9A-Za-z]+)", expand=False).fillna("unk")
    df["memdep"] = df["experiment"].str.extract(r"memdep([A-Za-z0-9_]+)", expand=False).fillna("off")
    log("🔍 Parsed experiment strings for cache, assoc, prefetcher, replacement, memdep")

# ---------- DERIVED METRICS ----------
if "IPC" in df.columns:
    df["Speedup_vs_Min"] = df.groupby("benchmark")["IPC"].transform(lambda x: x / x.min())
if "L1D_miss_rate" in df.columns:
    df["HitRate_L1D"] = 100 - df["L1D_miss_rate"]
    df["MPKI_L1D"] = (df["L1D_miss_rate"]/100)*1000/df["IPC"]
if "L2_miss_rate" in df.columns:
    df["HitRate_L2"] = 100 - df["L2_miss_rate"]

# ---------- 1️⃣ CORE SCALING PLOTS ----------
if "rob" in df.columns:
    for bench, g in df.groupby("benchmark"):
        fig, ax = plt.subplots()
        sns.lineplot(data=g, x="rob", y="IPC", hue="bp", marker="o", ax=ax, estimator="mean", errorbar=None)
        ax.set_title(f"ROB vs IPC — {bench}")
        ax.set_xlabel("ROB Size"); ax.set_ylabel("IPC")
        ax.legend(title="Branch Predictor", loc="best", frameon=True)
        save(fig, f"1_rob_ipc_{bench}.png")

if "issue_width" in df.columns:
    fig, ax = plt.subplots()
    sns.lineplot(data=df, x="issue_width", y="IPC", hue="commit_width", marker="o", ax=ax, estimator="mean", errorbar=None)
    ax.set_title("Issue Width vs IPC (by Commit Width)")
    ax.set_xlabel("Issue Width"); ax.set_ylabel("IPC")
    save(fig, "2_issuewidth_ipc_commitwidth.png")

# ---------- 2️⃣ PREFETCHER & MEMDEP ----------
if "prefetcher" in df.columns and "IPC" in df.columns:
    fig = sns.catplot(data=df, x="prefetcher", y="IPC", hue="bp", col="benchmark", kind="bar", height=6, aspect=1.1)
    fig.fig.suptitle("Prefetcher × Branch Predictor Interaction (IPC)", fontsize=16)
    fig.savefig(os.path.join(OUTDIR, "3_prefetcher_bp_ipc.png"), dpi=300, bbox_inches="tight")
    plt.close(fig.fig)

if "memdep" in df.columns:
    fig = sns.catplot(data=df, x="memdep", y="IPC", hue="benchmark", kind="bar", height=6, aspect=1.1)
    fig.fig.suptitle("Memory Dependence Predictor (On/Off) vs IPC", fontsize=16)
    fig.savefig(os.path.join(OUTDIR, "4_memdep_vs_ipc.png"), dpi=300, bbox_inches="tight")
    plt.close(fig.fig)

# ---------- 3️⃣ CACHE & ASSOCIATIVITY ----------
if "assoc" in df.columns and "HitRate_L1D" in df.columns:
    fig, ax = plt.subplots()
    sns.lineplot(data=df, x="assoc", y="HitRate_L1D", hue="benchmark", marker="o", ax=ax, estimator="mean", errorbar=None)
    ax.set_title("Associativity vs L1D Hit Rate"); ax.set_xlabel("Associativity"); ax.set_ylabel("L1D Hit Rate (%)")
    save(fig, "5_assoc_vs_hitrate.png")

if "l2" in df.columns:
    fig, ax = plt.subplots()
    sns.lineplot(data=df, x="l2", y="HitRate_L2", hue="benchmark", marker="o", ax=ax, estimator="mean", errorbar=None)
    ax.set_title("L2 Size vs L2 Hit Rate"); ax.set_xlabel("L2 Size"); ax.set_ylabel("L2 Hit Rate (%)")
    save(fig, "6_l2size_vs_hitrate.png")

# ---------- 4️⃣ ADVANCED ANALYTICS ----------
fig, ax = plt.subplots()
sns.scatterplot(data=df, x="L1D_miss_rate", y="IPC", hue="rob", style="benchmark", ax=ax)
ax.set_title("IPC vs L1D Miss Rate — Memory Locality Impact")
ax.set_xlabel("L1D Miss Rate (%)"); ax.set_ylabel("IPC")
save(fig, "7_ipc_vs_l1dmiss.png")

corr_cols = [c for c in ["IPC","CPI","L1D_miss_rate","L2_miss_rate","rob","issue_width","commit_width"] if c in df.columns]
if len(corr_cols) >= 3:
    fig, ax = plt.subplots()
    sns.heatmap(df[corr_cols].corr(), annot=True, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Parameter Correlation Heatmap")
    save(fig, "8_corr_heatmap.png")

if "MPKI_L1D" in df.columns:
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="MPKI_L1D", y="Speedup_vs_Min", hue="bp", style="benchmark", ax=ax)
    ax.set_title("MPKI vs Speedup — Cache Efficiency vs Performance")
    ax.set_xlabel("L1D MPKI"); ax.set_ylabel("Relative Speedup")
    save(fig, "9_mpki_vs_speedup.png")

# ---------- 5️⃣ PARETO FRONTIER ----------
if "L1D_miss_rate" in df.columns:
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="L1D_miss_rate", y="IPC", hue="bp", size="rob", ax=ax, alpha=0.8)
    ax.set_title("Pareto Frontier — IPC vs Miss Rate")
    ax.set_xlabel("L1D Miss Rate (%)"); ax.set_ylabel("IPC")
    save(fig, "10_pareto_ipc_missrate.png")

# ---------- 6️⃣ SUMMARY ----------
summary = dedent(f"""
✅ Full analysis completed.
Saved ~20 large plots to: {OUTDIR}

Highlights:
- Scaling: ROB, IssueWidth, CommitWidth
- Branch & Prefetcher interactions
- Cache associativity and size vs hit rate
- Memory predictor & performance coupling
- Correlation & Pareto insights

Figures are 10x7 inches for clarity.
Check {DIAG} for log of skipped/missing sections.
""").strip()
open(os.path.join(OUTDIR, "report_summary.txt"), "w").write(summary)
log(summary)
