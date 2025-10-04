#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plots_full_study_v2.py

Final high-clarity, annotated plotting suite for gem5 A3.
- Vivid palette
- Numeric annotations on bars and point markers
- Large, report-ready figures
- Graceful skipping and diagnostics
"""

import os
import sys
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from textwrap import dedent
from pandas.plotting import parallel_coordinates

# ---------- Config ----------
IN = "results/experiments_summary.csv"
OUTDIR = "analysis/plots_full_study_v2"
os.makedirs(OUTDIR, exist_ok=True)
DIAG = os.path.join(OUTDIR, "diagnostics.txt")

# Visual defaults (vivid + large)
sns.set_theme(style="whitegrid", font_scale=1.3)
plt.rcParams.update({
    "figure.figsize": (11, 7),
    "savefig.dpi": 300,
    "axes.titlesize": 18,
    "axes.labelsize": 14,
    "legend.fontsize": 12
})
PALETTE = "bright"  # vivid palette

# ---------- Helpers ----------
def log(msg):
    with open(DIAG, "a") as f:
        f.write(msg + "\n")
    print(msg)

def save_and_close(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved: {path}")

def enough_var(s, n=2):
    return s.dropna().nunique() >= n

def annotate_lineplot_points(ax, fmt="{:.3f}", offset=(0,6)):
    """Annotate points drawn on a lineplot. Expects markers present."""
    # iterate through Line2D objects
    for line in ax.get_lines():
        xdata = line.get_xdata()
        ydata = line.get_ydata()
        # annotate each visible point
        for x, y in zip(xdata, ydata):
            if pd.isna(y): continue
            ax.annotate(fmt.format(y), xy=(x, y), xytext=(0, offset[1]),
                        textcoords="offset points", ha="center", va="bottom", fontsize=11)

def annotate_bars(ax, fmt="{:.3f}", padding=3):
    """Annotate bars for barplot containers"""
    # matplotlib creates containers for bars
    for container in getattr(ax, "containers", []):
        try:
            ax.bar_label(container, fmt=fmt, padding=padding, fontsize=11)
        except Exception:
            pass

# ---------- Read & clean ----------
if not os.path.exists(IN):
    sys.exit(f"ERROR: Input file not found: {IN}")

df = pd.read_csv(IN)
orig = len(df)
if "status" in df.columns:
    df = df[df["status"].astype(str).str.lower() == "ok"].copy()
log(f"Loaded {orig} rows, {len(df)} rows with status==ok")

# strip headers
df.columns = [c.strip() for c in df.columns]

# numeric conversions
for c in ["IPC", "CPI", "L1D_miss_rate", "L2_miss_rate", "simInsts"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# convert miss rates to percent if they look like 0..1
for miss in ("L1D_miss_rate", "L2_miss_rate"):
    if miss in df.columns:
        mx = df[miss].max(skipna=True)
        if pd.notna(mx) and mx <= 1.0:
            df[miss] = df[miss] * 100.0
            log(f"Converted {miss} fraction -> percent")

# fill missing key columns with 'unknown' to avoid KeyErrors
key_cols = ["rob","bp","l1d","l1i","l2","issue_width","commit_width","experiment","benchmark"]
for k in key_cols:
    if k not in df.columns:
        df[k] = "unknown"

# ---------- auto-parse experiment tokens ----------
if "experiment" in df.columns:
    df["experiment"] = df["experiment"].astype(str)
    df["prefetcher"] = df["experiment"].str.extract(r"prefetcher([A-Za-z0-9_]+)", expand=False).fillna("none")
    df["replacement"] = df["experiment"].str.extract(r"(LRU|FIFO|Random|PLRU|MRU)", expand=False).fillna("default")
    df["assoc"] = df["experiment"].str.extract(r"assoc([0-9]+)", expand=False).fillna("unk")
    for lvl in ["l1d","l2"]:
        if lvl not in df.columns or (df[lvl].isna() | (df[lvl]=="unknown")).all():
            df[lvl] = df["experiment"].str.extract(rf"{lvl}[_-]?([0-9A-Za-z]+)", expand=False).fillna("unk")
    df["memdep"] = df["experiment"].str.extract(r"memdep([A-Za-z0-9_]+)", expand=False).fillna("off")
    log("Parsed experiment tokens: prefetcher, replacement, assoc, l1d/l2, memdep")

# ---------- derived metrics ----------
if "IPC" in df.columns:
    df["Speedup_vs_Min"] = df.groupby("benchmark")["IPC"].transform(lambda x: x / x.min())
if "L1D_miss_rate" in df.columns:
    df["HitRate_L1D"] = 100 - df["L1D_miss_rate"]
    df["MPKI_L1D"] = (df["L1D_miss_rate"]/100.0)*1000.0/df["IPC"]
if "L2_miss_rate" in df.columns:
    df["HitRate_L2"] = 100 - df["L2_miss_rate"]

# ---------- Plot: ROB vs IPC (per benchmark) ----------
if "rob" in df.columns and "IPC" in df.columns:
    for bench, g in df.groupby("benchmark"):
        if not (enough_var(g["rob"]) and enough_var(g["IPC"])):
            log(f"Skipping ROB vs IPC for {bench}: insufficient variance")
            continue
        fig, ax = plt.subplots(figsize=(12,7))
        sns.lineplot(data=g, x="rob", y="IPC", hue="bp", marker="o",
                     estimator="mean", errorbar=None, palette=PALETTE, ax=ax)
        ax.set_title(f"ROB vs IPC — {bench}")
        ax.set_xlabel("ROB Size")
        ax.set_ylabel("IPC")
        # annotate points
        annotate_lineplot_points(ax, fmt="{:.3f}", offset=(0,8))
        # move legend outside
        ax.legend(title="Branch Predictor", bbox_to_anchor=(1.02, 1), loc="upper left")
        save_and_close(fig, f"rob_vs_ipc_{bench}.png")
else:
    log("Skipping ROB vs IPC: missing columns")

# ---------- Plot: Issue width vs IPC (global) and per-benchmark ----------
if "issue_width" in df.columns and "IPC" in df.columns:
    # global
    fig, ax = plt.subplots(figsize=(12,7))
    sns.lineplot(data=df, x="issue_width", y="IPC", hue="commit_width", marker="o",
                 estimator="mean", errorbar=None, palette=PALETTE, ax=ax)
    ax.set_title("Issue Width vs IPC (by Commit Width)")
    ax.set_xlabel("Issue Width")
    ax.set_ylabel("IPC")
    annotate_lineplot_points(ax, fmt="{:.3f}", offset=(0,8))
    ax.legend(title="Commit Width", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_and_close(fig, "issuewidth_vs_ipc_commitwidth.png")

    # per-benchmark
    for bench, g in df.groupby("benchmark"):
        if not (enough_var(g["issue_width"]) and enough_var(g["IPC"])):
            log(f"Skipping per-benchmark IssueWidth vs IPC for {bench}: insufficient variance")
            continue
        fig, ax = plt.subplots(figsize=(10,6))
        sns.lineplot(data=g, x="issue_width", y="IPC", hue="commit_width", marker="o",
                     estimator="mean", errorbar=None, palette=PALETTE, ax=ax)
        ax.set_title(f"Issue Width vs IPC — {bench}")
        ax.set_xlabel("Issue Width"); ax.set_ylabel("IPC")
        annotate_lineplot_points(ax, fmt="{:.3f}", offset=(0,8))
        ax.legend(title="Commit Width", bbox_to_anchor=(1.02, 1), loc="upper left")
        save_and_close(fig, f"issuewidth_vs_ipc_{bench}.png")
else:
    log("Skipping IssueWidth vs IPC: missing columns")

# ---------- Plot: Prefetcher × BP Interaction (faceted per benchmark) ----------
if "prefetcher" in df.columns and enough_var(df["prefetcher"]) and "IPC" in df.columns:
    try:
        g = sns.catplot(data=df, x="prefetcher", y="IPC", hue="bp", col="benchmark",
                        kind="bar", palette=PALETTE, height=6, aspect=1.1)
        g.fig.suptitle("Prefetcher × Branch Predictor Interaction (IPC)", fontsize=18, y=1.02)
        # annotate each bar in each axis
        for ax in g.axes.flat:
            annotate_bars(ax, fmt="{:.3f}", padding=4)
            ax.set_xlabel("Prefetcher")
            ax.set_ylabel("IPC")
            ax.legend().set_title("BP")
        g.fig.savefig(os.path.join(OUTDIR, "prefetcher_bp_interaction.png"), dpi=300, bbox_inches="tight")
        plt.close(g.fig)
        log("Saved: pref./bp interaction (faceted)")
    except Exception as e:
        log(f"Failed to draw prefetcher × bp interaction: {e}")
else:
    log("Skipping Prefetcher × BP: missing or constant prefetcher/IPC")

# ---------- Plot: MemDep predictor (on/off) vs IPC ----------
if "memdep" in df.columns and "IPC" in df.columns:
    try:
        fig = sns.catplot(data=df, x="memdep", y="IPC", hue="benchmark", kind="bar", palette=PALETTE, height=6, aspect=1.2)
        fig.fig.suptitle("Memory Dependence Predictor (On/Off) vs IPC", fontsize=18, y=1.02)
        for ax in fig.axes.flat:
            annotate_bars(ax, fmt="{:.3f}", padding=4)
        fig.fig.savefig(os.path.join(OUTDIR, "memdep_vs_ipc.png"), dpi=300, bbox_inches="tight")
        plt.close(fig.fig)
        log("Saved: memdep vs ipc")
    except Exception as e:
        log(f"Failed memdep plot: {e}")
else:
    log("Skipping memdep vs IPC: missing memdep/IPC")

# ---------- Plot: Associativity vs L1D Hit Rate ----------
if "assoc" in df.columns and "HitRate_L1D" in df.columns:
    if enough_var(df["assoc"]) and enough_var(df["HitRate_L1D"]):
        fig, ax = plt.subplots(figsize=(11,7))
        sns.lineplot(data=df, x="assoc", y="HitRate_L1D", hue="benchmark", marker="o",
                     estimator="mean", errorbar=None, palette=PALETTE, ax=ax)
        ax.set_title("Associativity vs L1D Hit Rate")
        ax.set_xlabel("Associativity"); ax.set_ylabel("L1D Hit Rate (%)")
        annotate_lineplot_points(ax, fmt="{:.2f}", offset=(0,6))
        ax.legend(title="Benchmark", bbox_to_anchor=(1.02, 1), loc="upper left")
        save_and_close(fig, "assoc_vs_l1d_hitrate.png")
    else:
        log("Skipping assoc vs hitrate: insufficient variance")
else:
    log("Skipping assoc vs hitrate: missing assoc or HitRate_L1D")

# ---------- Plot: L2 Size vs L2 Hit Rate ----------
if "l2" in df.columns and "HitRate_L2" in df.columns:
    if enough_var(df["l2"]) and enough_var(df["HitRate_L2"]):
        fig, ax = plt.subplots(figsize=(11,7))
        sns.lineplot(data=df, x="l2", y="HitRate_L2", hue="benchmark", marker="o",
                     estimator="mean", errorbar=None, palette=PALETTE, ax=ax)
        ax.set_title("L2 Size vs L2 Hit Rate")
        ax.set_xlabel("L2 Size"); ax.set_ylabel("L2 Hit Rate (%)")
        annotate_lineplot_points(ax, fmt="{:.2f}", offset=(0,6))
        ax.legend(title="Benchmark", bbox_to_anchor=(1.02,1), loc="upper left")
        save_and_close(fig, "l2size_vs_l2_hitrate.png")
    else:
        log("Skipping l2 size vs hitrate: insufficient variance")
else:
    log("Skipping l2 size vs hitrate: missing l2/HitRate_L2")

# ---------- Plot: IPC vs L1D miss rate (scatter) with ROB size as marker size/color -->
if "L1D_miss_rate" in df.columns and "IPC" in df.columns:
    fig, ax = plt.subplots(figsize=(12,8))
    # point size mapping from rob if numeric
    if pd.api.types.is_numeric_dtype(df["rob"]):
        sizes = (df["rob"] - df["rob"].min()) / max(1, (df["rob"].max() - df["rob"].min())) * 200 + 50
    else:
        sizes = 80
    sns.scatterplot(data=df, x="L1D_miss_rate", y="IPC", hue="rob", size=sizes,
                    style="benchmark" if "benchmark" in df.columns else None,
                    palette=PALETTE, ax=ax, legend="full")
    ax.set_title("IPC vs L1D Miss Rate — Memory Locality Impact")
    ax.set_xlabel("L1D Miss Rate (%)"); ax.set_ylabel("IPC")
    ax.legend(bbox_to_anchor=(1.02,1), loc="upper left")
    save_and_close(fig, "ipc_vs_l1d_missrate_scatter.png")
else:
    log("Skipping IPC vs L1D miss rate scatter: missing columns")

# ---------- Pareto-ish scatter: IPC vs L1D Miss Rate colored by BP, sized by ROB ----------
if "IPC" in df.columns and "L1D_miss_rate" in df.columns and "bp" in df.columns:
    fig, ax = plt.subplots(figsize=(12,8))
    s = None
    if pd.api.types.is_numeric_dtype(df["rob"]):
        s = (df["rob"].astype(float) - df["rob"].astype(float).min()) / max(1,(df["rob"].astype(float).max()-df["rob"].astype(float).min())) * 200 + 40
    sns.scatterplot(data=df, x="L1D_miss_rate", y="IPC", hue="bp", size=s, palette=PALETTE, ax=ax, alpha=0.85)
    ax.set_title("Pareto View: IPC vs L1D Miss Rate (size ~ ROB)")
    ax.set_xlabel("L1D Miss Rate (%)"); ax.set_ylabel("IPC")
    ax.legend(bbox_to_anchor=(1.02,1), loc="upper left")
    save_and_close(fig, "pareto_ipc_miss_rob.png")
else:
    log("Skipping pareto scatter: missing columns")

# ---------- Correlation heatmap ----------
corr_cols = [c for c in ["IPC","CPI","L1D_miss_rate","L2_miss_rate","rob","issue_width","commit_width"] if c in df.columns]
if len(corr_cols) >= 3:
    corr = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0, ax=ax, fmt=".2f")
    ax.set_title("Parameter Correlation Heatmap")
    save_and_close(fig, "parameter_correlation_heatmap.png")
else:
    log("Skipping correlation heatmap: not enough numeric columns")

# ---------- Parallel coordinates (design-space overview) ----------
sel_cols = [c for c in ["rob","issue_width","commit_width","IPC"] if c in df.columns]
if len(sel_cols) == 4:
    try:
        tmp = df[sel_cols].dropna().copy()
        tmp["rob"] = tmp["rob"].astype(str)  # class col must be categorical/string
        tmp["issue_width"] = pd.to_numeric(tmp["issue_width"], errors="coerce")
        tmp["commit_width"] = pd.to_numeric(tmp["commit_width"], errors="coerce")
        tmp["IPC"] = pd.to_numeric(tmp["IPC"], errors="coerce")
        fig = plt.figure(figsize=(12,8))
        parallel_coordinates(tmp, class_column="rob", colormap=plt.cm.get_cmap("tab10"))
        plt.title("Design Space Overview (ROB–Issue–Commit–IPC)")
        save_and_close(fig, "parallel_coordinates_designspace.png")
    except Exception as e:
        log(f"Skipping parallel coordinates due to: {e}")
else:
    log("Skipping parallel coordinates: missing columns")

# ---------- Final summary and report ----------
summary = dedent(f"""
Full study finished.
Input rows: {orig}
Valid (status==ok) rows: {len(df)}
Output directory: {OUTDIR}
Check {DIAG} for any skipped plots reasons.

Generated:
- ROB vs IPC (per benchmark)
- IssueWidth vs IPC (global + per-benchmark)
- Prefetcher × BP interaction (faceted)
- MemDep on/off vs IPC
- Associativity/L2 size vs hit rate
- IPC vs L1D miss rate scatter and Pareto-like scatter
- Correlation heatmap, parallel coords (if data present)

Annotations: numeric values on bars and points; vivid palette.
""").strip()

open(os.path.join(OUTDIR, "report_summary.txt"), "w").write(summary + "\n")
log(summary)
