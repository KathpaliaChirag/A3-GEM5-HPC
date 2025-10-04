#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plots_advanced_v3.py

Comprehensive, fixed and expanded visualization suite for gem5 A3 results.

Features:
- Fixes artifacts: bubble-size legend (no longer cryptic), memdep parsing, missing single-point line groups.
- Produces BOTH line and bar variants for each metric comparison (ROB, IssueWidth, CommitWidth, Prefetcher, MemDep, Assoc, L2 size).
- Annotates numeric values on bars and points.
- Adds extra analytics: Speedup vs baseline (bp==none), Pareto-like scatter, pairwise plots, correlation heatmap.
- Saves results to `analysis/plots_v3/` and logs to diagnostics.txt.

Author: CK + GPT-5 Thinking mini
"""

import os, sys, math
from textwrap import dedent
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

# ---------------- CONFIG ----------------
INPUT = "results/experiments_summary.csv"
OUTDIR = "analysis/plots_v3"
os.makedirs(OUTDIR, exist_ok=True)
DIAG = os.path.join(OUTDIR, "diagnostics.txt")

# Visual defaults (vivid + big)
sns.set_theme(style="whitegrid", font_scale=1.35)
plt.rcParams.update({
    "figure.figsize": (12, 8),
    "savefig.dpi": 300,
    "axes.titlesize": 20,
    "axes.labelsize": 15,
    "legend.fontsize": 12
})
PALETTE = "bright"  # vivid palette

# ---------------- HELPERS ----------------
def log(msg):
    with open(DIAG, "a") as f:
        f.write(msg + "\n")
    print(msg)

def save(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved: {path}")

def enough_var(s, n=2):
    return s.dropna().nunique() >= n

def annotate_line_points(ax, fmt="{:.3f}", offset=(0,7)):
    """Annotate marker points on line plots; use mean for aggregated lines."""
    for line in ax.get_lines():
        xdata = line.get_xdata()
        ydata = line.get_ydata()
        # if xdata are categorical strings, keep as-is
        for x, y in zip(xdata, ydata):
            if pd.isna(y): 
                continue
            ax.annotate(fmt.format(y), xy=(x, y), xytext=(0, offset[1]),
                        textcoords="offset points", ha="center", va="bottom", fontsize=11)

def annotate_bars(ax, fmt="{:.3f}", padding=3):
    """Annotate all bar containers with values."""
    for container in getattr(ax, "containers", []):
        try:
            ax.bar_label(container, fmt=fmt, padding=padding, fontsize=11)
        except Exception:
            pass

def mean_and_std_label(series, fmt="{:.3f} ± {:+.3f}"):
    m = series.mean()
    s = series.std()
    return fmt.format(m, s)

# ---------------- LOAD & CLEAN ----------------
if not os.path.exists(INPUT):
    sys.exit(f"ERROR: Input not found: {INPUT}")

df = pd.read_csv(INPUT)
orig_rows = len(df)
if "status" in df.columns:
    df = df[df["status"].astype(str).str.lower() == "ok"].copy()
log(f"Loaded {orig_rows} rows; {len(df)} rows with status == ok")

# normalize column names
df.columns = [c.strip() for c in df.columns]

# numeric conversions and miss-rate -> percent fix
for c in ["IPC", "CPI", "L1D_miss_rate", "L2_miss_rate", "simInsts"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

for miss in ("L1D_miss_rate", "L2_miss_rate"):
    if miss in df.columns:
        mx = df[miss].max(skipna=True)
        if pd.notna(mx) and mx <= 1.0:
            df[miss] = df[miss] * 100.0
            log(f"Converted {miss} fraction -> percent")

# ensure key columns exist
for k in ["rob","bp","l1d","l1i","l2","issue_width","commit_width","experiment","benchmark"]:
    if k not in df.columns:
        df[k] = np.nan

# ---------------- PARSE EXPERIMENT TOKENS ----------------
if "experiment" in df.columns:
    df["experiment"] = df["experiment"].astype(str)
    df["prefetcher"] = df["experiment"].str.extract(r"prefetcher([A-Za-z0-9_]+)", expand=False).fillna("none")
    df["replacement"] = df["experiment"].str.extract(r"(LRU|FIFO|Random|PLRU|MRU)", expand=False).fillna("default")
    df["assoc"] = df["experiment"].str.extract(r"assoc([0-9]+)", expand=False).fillna(np.nan)
    for lvl in ["l1d","l2"]:
        if lvl not in df.columns or (df[lvl].isna() | (df[lvl]=="unknown")).all():
            df[lvl] = df["experiment"].str.extract(rf"{lvl}[_-]?([0-9A-Za-z]+)", expand=False).fillna(np.nan)
    df["memdep"] = df["experiment"].str.extract(r"memdep([A-Za-z0-9_]+)", expand=False).fillna("off")
    log("Parsed experiment tokens: prefetcher, replacement, assoc, l1d/l2, memdep")

# derived metrics
if "IPC" in df.columns:
    df["Speedup_vs_Min"] = df.groupby("benchmark")["IPC"].transform(lambda x: x / x.min())
if "L1D_miss_rate" in df.columns:
    df["HitRate_L1D"] = 100.0 - df["L1D_miss_rate"]
    df["MPKI_L1D"] = (df["L1D_miss_rate"]/100.0)*1000.0/df["IPC"]
if "L2_miss_rate" in df.columns:
    df["HitRate_L2"] = 100.0 - df["L2_miss_rate"]

# cast rob to numeric if possible
try:
    df["rob"] = pd.to_numeric(df["rob"], errors="coerce")
except Exception:
    pass

# ---------------- HELPFUL FIXES (for problems you reported) ----------------
# 1) remove cryptic scatter 'size' ticks by creating a custom ROB-size legend for bubble plots
def add_rob_legend(ax, rob_sizes, marker='o'):
    # rob_sizes: sorted unique numeric ROBs we want to represent
    from matplotlib.lines import Line2D
    handles = []
    labels = []
    for r in rob_sizes:
        size = 80 + ( (float(r) - float(min(rob_sizes))) / max(1,(float(max(rob_sizes))-float(min(rob_sizes)))) ) * 180
        handles.append(Line2D([0],[0], marker=marker, color='w', markerfacecolor='gray', markersize=math.sqrt(size), markeredgecolor='k', linestyle=''))
        labels.append(str(int(r)))
    ax.legend(handles, labels, title="ROB size", bbox_to_anchor=(1.02, 0.5), loc="center left")
    return ax

# 2) Ensure IssueWidth lines show even when a group has only one data point:
# We'll compute grouped means and plot them as lines (so line connects group means)
def grouped_means(df, groupby_cols, xcol, ycol):
    g = df.groupby(groupby_cols)[ycol].mean().reset_index()
    return g

# ---------------- PLOTTING - produce BOTH line and bar variants ----------------
# Helpers to produce both charts for a given grouping
def plot_line_and_bar(df, x, y, hue=None, col=None, title=None, fname_prefix=None, order=None, hue_order=None, aggfunc='mean', annotate_fmt="{:.3f}"):
    """Create a line (with markers) and a bar plot for same aggregation."""
    if fname_prefix is None:
        fname_prefix = f"{x}_vs_{y}"
    # prepare aggregated data (mean) to ensure lines even with sparse groups
    group_cols = [c for c in ([x] + ([hue] if hue else []) + ([col] if col else [])) if c]
    agg = df.groupby(group_cols)[y].agg(['mean','std','count']).reset_index().rename(columns={'mean':y+"_mean",'std':y+"_std",'count':y+"_count"})
    # convert x to ordered if order provided
    if order is not None:
        agg[x] = pd.Categorical(agg[x], categories=order, ordered=True)
        agg = agg.sort_values(x)
    # ---------------- LINE ----------------
    fig, ax = plt.subplots(figsize=(12,8))
    if hue:
        # plot each hue group as a line over x from agg
        hue_vals = agg[hue].unique() if hue_order is None else hue_order
        for h in hue_vals:
            sub = agg[agg[hue]==h]
            if sub.empty: continue
            sns.lineplot(data=sub, x=x, y=y+"_mean", marker="o", ax=ax, label=str(h))
            # annotate points
            for xi, yi in zip(sub[x], sub[y+"_mean"]):
                if pd.isna(yi): continue
                ax.annotate(annotate_fmt.format(yi), xy=(xi, yi), xytext=(0,7), textcoords="offset points", ha="center", va="bottom", fontsize=11)
    else:
        sns.lineplot(data=agg, x=x, y=y+"_mean", marker="o", ax=ax)
        for xi, yi in zip(agg[x], agg[y+"_mean"]):
            if pd.isna(yi): continue
            ax.annotate(annotate_fmt.format(yi), xy=(xi, yi), xytext=(0,7), textcoords="offset points", ha="center", va="bottom", fontsize=11)
    ax.set_title((title or f"{x} vs {y} (line)"))
    ax.set_xlabel(x); ax.set_ylabel(y)
    ax.legend(title=hue if hue else None, bbox_to_anchor=(1.02, 1), loc="upper left")
    save(fig, f"{fname_prefix}_line.png")

    # ---------------- BAR ----------------
    fig, ax = plt.subplots(figsize=(12,8))
    if hue and col:
        # use seaborn catplot for faceted bars and annotate each sub-axis
        g = sns.catplot(data=df, x=x, y=y, hue=hue, col=col, kind="bar", palette=PALETTE, height=6, aspect=1.1)
        g.fig.suptitle(title or f"{x} vs {y} (bar, faceted)", fontsize=18, y=1.02)
        for subax in g.axes.flat:
            annotate_bars(subax, fmt=annotate_fmt, padding=4)
        g.fig.savefig(os.path.join(OUTDIR, f"{fname_prefix}_bar_faceted.png"), dpi=300, bbox_inches="tight")
        plt.close(g.fig)
        log(f"Saved faceted bar: {fname_prefix}_bar_faceted.png")
    elif hue:
        sns.barplot(data=df, x=x, y=y, hue=hue, palette=PALETTE, estimator=np.mean, ci=None, ax=ax)
        annotate_bars(ax, fmt=annotate_fmt, padding=4)
        ax.set_title((title or f"{x} vs {y} (bar)"))
        ax.set_xlabel(x); ax.set_ylabel(y)
        ax.legend(title=hue, bbox_to_anchor=(1.02, 1), loc="upper left")
        save(fig, f"{fname_prefix}_bar.png")
    else:
        sns.barplot(data=df, x=x, y=y, palette=PALETTE, estimator=np.mean, ci=None, ax=ax)
        annotate_bars(ax, fmt=annotate_fmt, padding=4)
        ax.set_title((title or f"{x} vs {y} (bar)"))
        ax.set_xlabel(x); ax.set_ylabel(y)
        save(fig, f"{fname_prefix}_bar.png")

# ---------------- PRODUCE PLOTS ----------------

# 1) ROB vs IPC (line + bar) per benchmark and global
if "rob" in df.columns and "IPC" in df.columns:
    # global line + bar (hue=bp)
    plot_line_and_bar(df, x="rob", y="IPC", hue="bp", title="ROB vs IPC (global) - line and bar", fname_prefix="rob_ipc_global", annotate_fmt="{:.3f}")
    # per-benchmark
    for bench, sub in df.groupby("benchmark"):
        if not enough_var(sub["rob"]) or not enough_var(sub["IPC"]):
            log(f"Skipping ROB vs IPC for benchmark={bench}: insufficient variance")
            continue
        plot_line_and_bar(sub, x="rob", y="IPC", hue="bp", title=f"ROB vs IPC — {bench}", fname_prefix=f"rob_ipc_{bench}", annotate_fmt="{:.3f}")
else:
    log("Skipping ROB vs IPC: columns missing")

# 2) Issue width vs IPC and CPI - both line and bar, per benchmark and global
if "issue_width" in df.columns and "IPC" in df.columns:
    plot_line_and_bar(df, x="issue_width", y="IPC", hue="commit_width", title="Issue Width vs IPC (global) - line and bar", fname_prefix="issuewidth_ipc_global", annotate_fmt="{:.3f}")
    plot_line_and_bar(df, x="issue_width", y="CPI", hue="commit_width", title="Issue Width vs CPI (global) - line and bar", fname_prefix="issuewidth_cpi_global", annotate_fmt="{:.3f}")

    for bench, sub in df.groupby("benchmark"):
        if not enough_var(sub["issue_width"]) or not enough_var(sub["IPC"]):
            log(f"Skipping issuewidth vs IPC for {bench}: insufficient variance")
            continue
        plot_line_and_bar(sub, x="issue_width", y="IPC", hue="commit_width", title=f"Issue Width vs IPC — {bench}", fname_prefix=f"issuewidth_ipc_{bench}", annotate_fmt="{:.3f}")
else:
    log("Skipping IssueWidth plots: missing columns")

# 3) Commit width vs IPC (line + bar)
if "commit_width" in df.columns and "IPC" in df.columns:
    plot_line_and_bar(df, x="commit_width", y="IPC", hue="issue_width", title="Commit Width vs IPC (global) - line and bar", fname_prefix="commitwidth_ipc_global", annotate_fmt="{:.3f}")
else:
    log("Skipping CommitWidth vs IPC: missing columns")

# 4) Prefetcher × BP Interaction (bar + line)
if "prefetcher" in df.columns and "IPC" in df.columns:
    # faceted bars by benchmark (already useful)
    plot_line_and_bar(df, x="prefetcher", y="IPC", hue="bp", col="benchmark", title="Prefetcher vs IPC (by BP and benchmark)", fname_prefix="prefetcher_ipc_bp", annotate_fmt="{:.3f}")
    # also line variant showing mean IPC per prefetcher across bps
    agg = df.groupby(["prefetcher","bp"])["IPC"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(12,8))
    sns.lineplot(data=agg, x="prefetcher", y="IPC", hue="bp", marker="o", palette=PALETTE, ax=ax)
    annotate_line_points(ax, fmt="{:.3f}")
    ax.set_title("Prefetcher vs IPC (line)")
    ax.set_xlabel("Prefetcher"); ax.set_ylabel("IPC")
    ax.legend(title="BP", bbox_to_anchor=(1.02,1), loc="upper left")
    save(fig, "prefetcher_ipc_line.png")
else:
    log("Skipping Prefetcher vs IPC: missing columns")

# 5) MemDep predictor (on/off) vs IPC - bar + line
if "memdep" in df.columns and "IPC" in df.columns:
    # ensure memdep has appropriate categories (on/off)
    df["memdep"] = df["memdep"].astype(str)
    # bar faceted by benchmark
    try:
        g = sns.catplot(data=df, x="memdep", y="IPC", hue="benchmark", kind="bar", palette=PALETTE, height=6, aspect=1.2)
        g.fig.suptitle("Memory Dependence Predictor (On/Off) vs IPC (bar)", fontsize=18, y=1.02)
        for ax in g.axes.flat:
            annotate_bars(ax, fmt="{:.3f}", padding=4)
        g.fig.savefig(os.path.join(OUTDIR, "memdep_ipc_bar.png"), dpi=300, bbox_inches="tight")
        plt.close(g.fig)
        log("Saved: memdep_ipc_bar.png")
    except Exception as e:
        log(f"Memdep bar failed: {e}")

    # line: mean IPC per memdep state
    try:
        agg = df.groupby(["memdep","benchmark"])["IPC"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(12,8))
        sns.lineplot(data=agg, x="memdep", y="IPC", hue="benchmark", marker="o", palette=PALETTE, ax=ax)
        annotate_line_points(ax, fmt="{:.3f}")
        ax.set_title("Memory Dependence Predictor (On/Off) vs IPC (line)")
        ax.set_xlabel("memdep"); ax.set_ylabel("IPC")
        ax.legend(title="benchmark", bbox_to_anchor=(1.02,1), loc="upper left")
        save(fig, "memdep_ipc_line.png")
    except Exception as e:
        log(f"Memdep line failed: {e}")
else:
    log("Skipping memdep plots: missing data")

# 6) Associativity & L1D HitRate (line + bar)
if "assoc" in df.columns and "HitRate_L1D" in df.columns:
    plot_line_and_bar(df, x="assoc", y="HitRate_L1D", hue="benchmark", title="Associativity vs L1D Hit Rate", fname_prefix="assoc_l1d_hitrate", annotate_fmt="{:.2f}")
else:
    log("Skipping assoc vs hit rate: missing columns")

# 7) L2 Size vs L2 hit rate (line + bar)
if "l2" in df.columns and "HitRate_L2" in df.columns:
    plot_line_and_bar(df, x="l2", y="HitRate_L2", hue="benchmark", title="L2 Size vs L2 Hit Rate", fname_prefix="l2_hitrate", annotate_fmt="{:.2f}")
else:
    log("Skipping L2 size vs hit rate: missing columns")

# 8) IPC vs L1D Miss Rate — scatter + Pareto, but fix bubble legend
if "L1D_miss_rate" in df.columns and "IPC" in df.columns:
    fig, ax = plt.subplots(figsize=(12,9))
    # choose a set of canonical ROB sizes to represent, if numeric
    if pd.api.types.is_numeric_dtype(df["rob"]):
        rob_unique = sorted(df["rob"].dropna().unique())
        # map numeric rob to marker sizes for points
        def rob_to_size(r):
            if math.isnan(r): return 60
            # scale sizes between 60 and 260
            rmin, rmax = min(rob_unique), max(rob_unique)
            if rmax==rmin: return 100
            return 60 + ( (r - rmin) / (rmax - rmin) ) * 200
        sizes = df["rob"].apply(lambda r: rob_to_size(r) if not pd.isna(r) else 80)
    else:
        sizes = 80
        rob_unique = []

    sns.scatterplot(data=df, x="L1D_miss_rate", y="IPC", hue="benchmark", style="benchmark", s=sizes, palette=PALETTE, alpha=0.85, ax=ax, legend="brief")
    ax.set_title("IPC vs L1D Miss Rate — Memory Locality Impact (bubble size ≈ ROB)")
    ax.set_xlabel("L1D Miss Rate (%)"); ax.set_ylabel("IPC")
    # custom ROB legend — if numeric rob present
    if rob_unique:
        add_rob_legend(ax, rob_unique[:3] if len(rob_unique)>3 else rob_unique)  # show up to 3 representative sizes
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    save(fig, "ipc_vs_l1dmiss_scatter.png")

    # Pareto-ish: color by BP, size by rob (same mapping)
    fig, ax = plt.subplots(figsize=(12,9))
    sns.scatterplot(data=df, x="L1D_miss_rate", y="IPC", hue="bp", size=df["rob"].fillna(df["rob"].median()), sizes=(60, 260), palette=PALETTE, alpha=0.85, ax=ax)
    ax.set_title("Pareto-like: IPC vs L1D Miss Rate (color=BP, size~ROB)")
    ax.set_xlabel("L1D Miss Rate (%)"); ax.set_ylabel("IPC")
    # make ROB legend explicit by creating handles (if numeric)
    if pd.api.types.is_numeric_dtype(df["rob"]):
        add_rob_legend(ax, rob_unique[:3] if len(rob_unique)>3 else rob_unique)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    save(fig, "pareto_ipc_l1dmiss.png")
else:
    log("Skipping IPC vs L1D miss scatter: missing columns")

# 9) MPKI vs Speedup (scatter) if available
if "MPKI_L1D" in df.columns and "Speedup_vs_Min" in df.columns:
    fig, ax = plt.subplots(figsize=(12,8))
    sns.scatterplot(data=df, x="MPKI_L1D", y="Speedup_vs_Min", hue="bp", palette=PALETTE, alpha=0.8, ax=ax)
    ax.set_title("L1D MPKI vs Speedup (relative to min config, per benchmark)")
    ax.set_xlabel("L1D MPKI"); ax.set_ylabel("Relative Speedup")
    save(fig, "mpki_vs_speedup.png")
else:
    log("Skipping MPKI vs Speedup: missing columns")

# 10) Correlation heatmap
corr_cols = [c for c in ["IPC","CPI","L1D_miss_rate","L2_miss_rate","rob","issue_width","commit_width"] if c in df.columns]
if len(corr_cols) >= 3:
    corr = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(11,9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Parameter Correlation Heatmap")
    save(fig, "corr_heatmap.png")
else:
    log("Skipping correlation heatmap: not enough numeric columns")

# 11) Parallel coordinates design-space (ROB as categorical)
sel_cols = [c for c in ["rob","issue_width","commit_width","IPC"] if c in df.columns]
if len(sel_cols) == 4:
    try:
        tmp = df[sel_cols].dropna().copy()
        tmp["rob"] = tmp["rob"].astype(str)
        tmp["issue_width"] = pd.to_numeric(tmp["issue_width"], errors="coerce")
        tmp["commit_width"] = pd.to_numeric(tmp["commit_width"], errors="coerce")
        tmp["IPC"] = pd.to_numeric(tmp["IPC"], errors="coerce")
        fig = plt.figure(figsize=(12,8))
        parallel_coordinates(tmp, class_column="rob", colormap=plt.cm.get_cmap("tab10"))
        plt.title("Design Space Overview (ROB–Issue–Commit–IPC)")
        save(fig, "parallel_coordinates_designspace.png")
    except Exception as e:
        log(f"Skipping parallel coordinates: {e}")
else:
    log("Skipping parallel coordinates: missing columns")

# 12) Speedup vs No-Predictor baseline (bp == none) heatmap
if "bp" in df.columns and "IPC" in df.columns:
    base = df[df["bp"].astype(str).str.lower()=="none"]
    if not base.empty:
        merged = df.merge(base[["benchmark","rob","issue_width","IPC"]], on=["benchmark","rob","issue_width"], suffixes=("","_none"))
        merged["Speedup_vs_NoneBP"] = merged["IPC"]/merged["IPC_none"]
        pivot = merged.pivot_table(values="Speedup_vs_NoneBP", index="rob", columns="bp", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(12,8))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax)
        ax.set_title("Speedup vs No-Predictor Baseline (avg over benchmarks)")
        save(fig, "speedup_vs_nonebp_heatmap.png")
    else:
        log("Skipping Speedup vs NoneBP: no 'none' branch predictor runs in data")
else:
    log("Skipping Speedup vs NoneBP: missing columns")

# ---------------- EXTRA: Save best config per benchmark (by IPC) ----------------
if "benchmark" in df.columns and "IPC" in df.columns:
    best = df.sort_values(["benchmark","IPC"], ascending=[True, False]).groupby("benchmark").first().reset_index()
    best_cols = ["benchmark","dir","experiment","IPC","CPI","rob","issue_width","commit_width","bp","prefetcher","l1d","l2"]
    best_out = best[[c for c in best_cols if c in best.columns]]
    best_out.to_csv(os.path.join(OUTDIR, "best_config_per_benchmark.csv"), index=False)
    log("Saved best_config_per_benchmark.csv")
else:
    log("Skipping best-config export: missing benchmark/IPC")

# ---------------- DIAGNOSTICS / SUMMARY ----------------
summary = dedent(f"""
plots_advanced_v3.py finished.
Input rows total: {orig_rows}
Rows used (status==ok): {len(df)}
Output directory: {OUTDIR}
Diagnostics: {DIAG}

Notes:
- We removed cryptic bubble-size ticks by adding an explicit ROB legend for bubble plots.
- IssueWidth lines are computed with grouped means to ensure visible lines even with single-point groups.
- Memdep parsed from experiment names; if only one state present ('off' or 'on') the other state won't draw a bar.
- Both line and bar variants are produced for primary comparisons (ROB, IssueWidth, CommitWidth, Prefetcher, Assoc, L2 size, MemDep).
""").strip()

open(os.path.join(OUTDIR, "report_summary.txt"), "w").write(summary + "\n")
log(summary)
