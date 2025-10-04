#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plots_metrics_explanatory.py — High-level, explanatory plots for gem5 A3 (HPC Project)

Generates 8 major performance plots + 1 summary table with best IPC config.
Author: CK + GPT-5
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------- Config ----------
IN = "results/experiments_summary.csv"
OUTDIR = "analysis/plots_explanatory"
os.makedirs(OUTDIR, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk", palette="muted")

# ---------- Load and Clean ----------
df = pd.read_csv(IN)
df = df[df["status"].astype(str).str.lower() == "ok"].copy()
print(f"✅ Loaded {len(df)} successful runs")

# Convert numeric columns
for col in ["IPC", "CPI", "L1D_miss_rate", "L2_miss_rate"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if "miss" in col and df[col].max(skipna=True) <= 1:
            df[col] *= 100

# ---------- Helper ----------
def save_plot(fig, name):
    fig.tight_layout()
    out = os.path.join(OUTDIR, name)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"📊 Saved: {out}")

def clean_labels(ax):
    ax.set_xlabel(ax.get_xlabel().replace("_", " ").title())
    ax.set_ylabel(ax.get_ylabel().replace("_", " ").title())
    ax.legend(title=ax.legend().get_title().get_text().replace("_", " ").title(), loc="best", fontsize=9)

# ---------- 1️⃣ ROB vs IPC (group by Issue width, per benchmark) ----------
for bench, group in df.groupby("benchmark"):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=group, x="rob", y="IPC", hue="issue_width", ax=ax)
    ax.set_title(f"ROB size vs IPC — {bench}")
    clean_labels(ax)
    save_plot(fig, f"rob_ipc_issuewidth_{bench}.png")

# ---------- 2️⃣ Issue width vs IPC/CPI (group by Commit width) ----------
for metric in ["IPC", "CPI"]:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df, x="issue_width", y=metric, hue="commit_width", ax=ax)
    ax.set_title(f"Issue width vs {metric} (grouped by Commit width)")
    clean_labels(ax)
    save_plot(fig, f"issuewidth_{metric.lower()}_commitwidth.png")

# ---------- 3️⃣ L1D size vs L1D miss rate (group by Prefetcher policy) ----------
if "L1D_miss_rate" in df.columns:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df, x="l1d", y="L1D_miss_rate", hue="bp", ax=ax)
    ax.set_title("L1D size vs L1D miss rate (Prefetcher policy impact)")
    clean_labels(ax)
    save_plot(fig, "l1d_missrate_prefetcher.png")

# ---------- 4️⃣ L2 size vs L2 miss rate (group by Associativity) ----------
if "L2_miss_rate" in df.columns and "l2" in df.columns:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df, x="l2", y="L2_miss_rate", hue="l1d", ax=ax)
    ax.set_title("L2 size vs L2 miss rate (Associativity effect)")
    clean_labels(ax)
    save_plot(fig, "l2_missrate_associativity.png")

# ---------- 5️⃣ Prefetcher vs IPC (group by Branch predictor) ----------
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=df, x="l1d", y="IPC", hue="bp", ax=ax)
ax.set_title("Prefetcher vs IPC (grouped by Branch Predictor)")
clean_labels(ax)
save_plot(fig, "prefetcher_ipc_bp.png")

# ---------- 6️⃣ Branch predictor vs IPC/CPI ----------
for metric in ["IPC", "CPI"]:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df, x="bp", y=metric, ax=ax)
    ax.set_title(f"Branch Predictor vs {metric}")
    clean_labels(ax)
    save_plot(fig, f"bp_{metric.lower()}.png")

# ---------- 7️⃣ Replacement policy vs IPC (grouped by cache size) ----------
if "l2" in df.columns:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df, x="l1d", y="IPC", hue="l2", ax=ax)
    ax.set_title("Replacement policy vs IPC (grouped by cache size)")
    clean_labels(ax)
    save_plot(fig, "replacement_ipc_cache.png")

# ---------- 8️⃣ Benchmark vs Failure rate (Prefetcher / ROB) ----------
fail = pd.read_csv(IN)
fail["failed"] = fail["status"].apply(lambda s: 0 if str(s).lower() == "ok" else 1)
fail_rate = fail.groupby(["benchmark", "l1d", "rob"])["failed"].mean().reset_index()

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=fail_rate, x="benchmark", y="failed", hue="l1d", ax=ax)
ax.set_title("Failure rate per benchmark (Prefetcher/ROB influence)")
ax.set_ylabel("Failure Rate (fraction)")
ax.set_xlabel("Benchmark")
ax.legend(title="L1D size")
plt.xticks(rotation=45, ha="right")
save_plot(fig, "failure_rate_benchmark.png")

# ---------- 9️⃣ Highest IPC per benchmark ----------
best_ipc = (
    df.loc[df.groupby("benchmark")["IPC"].idxmax()]
    [["benchmark", "experiment", "IPC", "rob", "bp", "l1d", "l2", "issue_width", "commit_width"]]
    .sort_values("benchmark")
)
best_ipc.to_csv(os.path.join(OUTDIR, "best_config_per_benchmark.csv"), index=False)
print("✅ Saved best IPC summary table: best_config_per_benchmark.csv")

print("\n🎉 All explanatory plots and summary generated successfully!")
