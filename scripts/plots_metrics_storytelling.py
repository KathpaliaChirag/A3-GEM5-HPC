#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plots_metrics_storytelling.py — Explanatory & Insightful gem5 HPC Analysis
Author: CK + GPT-5

Produces high-level, trend-based plots explaining architectural impacts.
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------- Config ----------
IN = "results/experiments_summary.csv"
OUTDIR = "analysis/plots_storytelling"
os.makedirs(OUTDIR, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk", palette="deep")

# ---------- Load & Clean ----------
df = pd.read_csv(IN)
df = df[df["status"].astype(str).str.lower() == "ok"].copy()
for col in ["IPC", "CPI", "L1D_miss_rate", "L2_miss_rate"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if "miss" in col and df[col].max(skipna=True) <= 1:
            df[col] *= 100

print(f"✅ Loaded {len(df)} successful runs")

# ---------- Helper ----------
def save(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"📈 Saved {path}")

# ---------- 1️⃣ ROB vs IPC (group by issue width, per benchmark) ----------
for bench, group in df.groupby("benchmark"):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.lineplot(data=group, x="rob", y="IPC", hue="issue_width", marker="o", ax=ax)
    ax.set_title(f"ROB Scaling — {bench}")
    ax.set_ylabel("Instructions per Cycle (IPC)")
    ax.set_xlabel("ROB Size")
    ax.legend(title="Issue Width", loc="best")
    save(fig, f"rob_ipc_{bench}.png")

# ---------- 2️⃣ Issue width vs IPC and CPI ----------
fig, ax1 = plt.subplots(figsize=(8, 5))
sns.lineplot(data=df, x="issue_width", y="IPC", hue="commit_width", marker="o", ax=ax1)
ax1.set_title("Issue Width Scaling — IPC & CPI")
ax1.set_ylabel("IPC")
ax1.set_xlabel("Issue Width")

ax2 = ax1.twinx()
sns.lineplot(data=df, x="issue_width", y="CPI", hue="commit_width", marker="s", linestyle="--", ax=ax2, legend=False)
ax2.set_ylabel("CPI")
save(fig, "issuewidth_ipc_cpi_commitwidth.png")

# ---------- 3️⃣ L1D size vs L1D miss rate (Prefetcher) ----------
fig, ax = plt.subplots(figsize=(8, 5))
sns.lineplot(data=df, x="l1d", y="L1D_miss_rate", hue="bp", marker="o", ax=ax)
ax.set_title("L1D Size vs Miss Rate — Prefetcher / BP Impact")
ax.set_ylabel("L1D Miss Rate (%)")
save(fig, "l1d_missrate_prefetcher.png")

# ---------- 4️⃣ L2 size vs L2 miss rate (Assoc) ----------
fig, ax = plt.subplots(figsize=(8, 5))
sns.lineplot(data=df, x="l2", y="L2_miss_rate", hue="l1d", marker="o", ax=ax)
ax.set_title("L2 Size vs Miss Rate — Associativity Impact")
ax.set_ylabel("L2 Miss Rate (%)")
save(fig, "l2_missrate_assoc.png")

# ---------- 5️⃣ Prefetcher vs IPC (group by BP) ----------
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=df, x="l1d", y="IPC", hue="bp", ax=ax)
ax.set_title("Prefetcher Type vs IPC (Grouped by BP)")
ax.set_ylabel("IPC")
save(fig, "prefetcher_ipc_bp.png")

# ---------- 6️⃣ Branch Predictor vs IPC/CPI ----------
for metric in ["IPC", "CPI"]:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=df, x="bp", y=metric, ax=ax)
    ax.set_title(f"Branch Predictor vs {metric}")
    ax.set_ylabel(metric)
    save(fig, f"bp_{metric.lower()}.png")

# ---------- 7️⃣ Replacement Policy Impact ----------
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=df, x="l2", y="IPC", hue="l1d", ax=ax)
ax.set_title("Cache Replacement Policy Impact (by Cache Size)")
ax.set_ylabel("IPC")
save(fig, "replacement_policy_ipc.png")

# ---------- 8️⃣ Failure rate heatmap ----------
fail = pd.read_csv(IN)
fail["failed"] = fail["status"].apply(lambda s: 0 if str(s).lower() == "ok" else 1)
fail_rate = fail.groupby(["benchmark", "rob"])["failed"].mean().unstack().fillna(0)

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(fail_rate, annot=True, cmap="Reds", fmt=".2f", ax=ax)
ax.set_title("Failure Rate by Benchmark & ROB Size")
ax.set_xlabel("ROB Size")
ax.set_ylabel("Benchmark")
save(fig, "failure_rate_heatmap.png")

# ---------- 9️⃣ Best IPC per Benchmark ----------
best = df.loc[df.groupby("benchmark")["IPC"].idxmax()]
best = best[["benchmark", "experiment", "IPC", "rob", "bp", "l1d", "l2", "issue_width", "commit_width"]]
best = best.sort_values("benchmark")
best.to_csv(os.path.join(OUTDIR, "best_config_summary.csv"), index=False)
print("✅ Saved best configuration summary table")

print("\n🎯 Done! Insightful plots and summary ready at analysis/plots_storytelling/")
