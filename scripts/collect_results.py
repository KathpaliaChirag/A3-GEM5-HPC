#!/usr/bin/env python3
"""
collect_results.py

Scan an experiments directory (default: results/experiments), parse non-empty stats.txt files,
extract a small set of metrics and parameters (parsed from folder name), and write a CSV summary.

Usage:
  python3 scripts/collect_results.py \
      --experiments-dir results/experiments \
      --output results/experiments_summary.csv
"""
import os
import re
import csv
import argparse
from pathlib import Path

# Candidate stat keys (we try these in order)
IPC_KEYS = ["system.cpu.ipc"]
CPI_KEYS = ["system.cpu.cpi"]
SIMINSTS_KEYS = ["simInsts", "simInsts::total"]
L1D_MISS_KEYS = [
    "system.cpu.dcache.demandMissRate::total",
    "system.cpu.dcache.overallMissRate::total",
    "system.cpu.dcache.demandMissRate::cpu.data",
    "system.cpu.dcache.overallMissRate::cpu.data",
    "system.cpu.dcache.demandMissRate",
    "system.cpu.dcache.overallMissRate",
]
L2_MISS_KEYS = [
    "system.l2cache.overallMissRate::total",
    "system.l2cache.demandMissRate::total",
    "system.l2cache.overallMissRate::cpu.data",
    "system.l2cache.demandMissRate::cpu.data",
    "system.l2cache.overallMissRate",
    "system.l2cache.demandMissRate",
]

KV_RE = re.compile(r'^\s*([^\s]+)\s+([+-]?[0-9]*\.?[0-9]+([eE][+-]?[0-9]+)?)')

def parse_stats_file(path):
    """Return dict of {stat_name: float} parsed from stats.txt"""
    stats = {}
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            m = KV_RE.match(line)
            if not m:
                continue
            key = m.group(1).strip()
            val_str = m.group(2)
            try:
                val = float(val_str)
            except ValueError:
                continue
            stats[key] = val
    return stats

def pick_first(stats, keys):
    for k in keys:
        if k in stats:
            return stats[k]
    return None

# Parse experiment directory name for parameters (best-effort)
def parse_experiment_name(name):
    # name examples:
    # compute_rob64_bpTournamentBP_l1d32kB_l2256kB_iw2_cw2
    res = {
        "experiment": name,
        "benchmark": None,
        "rob": None,
        "bp": None,
        "l1d": None,
        "l1i": None,
        "l2": None,
        "iw": None,
        "cw": None,
    }
    res["benchmark"] = name.split('_')[0] if '_' in name else name

    re_map = {
        "rob": r'rob(\d+)',
        "bp": r'bp([A-Za-z0-9]+)',
        "l1d": r'l1d(\d+(?:kB|KB|KiB)?)',
        "l1i": r'l1i(\d+(?:kB|KB|KiB)?)',
        # accept possibly missing underscore like "l2256kB" or "l2_256kB"
        "l2": r'l2_?(\d+(?:kB|KB|KiB)?)',
        "iw": r'iw(\d+)',
        "cw": r'cw(\d+)',
    }
    for k, pat in re_map.items():
        m = re.search(pat, name)
        if m:
            res[k] = m.group(1)
    # fallback: if no l1i found, try generic l1(\d+kB)
    if not res["l1i"]:
        m = re.search(r'l1(?:i|)\s*?(\d+(?:kB|KB|KiB)?)', name)
        if m:
            res["l1i"] = m.group(1)
    return res

def collect(experiments_dir, out_csv, verbose=False):
    experiments_dir = Path(experiments_dir)
    if not experiments_dir.exists():
        raise SystemExit(f"Experiments directory not found: {experiments_dir}")

    rows = []
    total = 0
    succeeded = 0
    failed = 0

    for entry in sorted(experiments_dir.iterdir()):
        if not entry.is_dir():
            continue
        total += 1
        stats_path = entry / "stats.txt"
        console_path = entry / "console.txt"
        status = "missing-stats"
        if stats_path.exists() and stats_path.stat().st_size > 0:
            try:
                stats = parse_stats_file(stats_path)
            except Exception as e:
                stats = {}
                status = f"parse-error:{e}"
                failed += 1
                if verbose:
                    print(f"[ERR] parsing {stats_path}: {e}")
            else:
                status = "ok"
                succeeded += 1
        else:
            # empty or missing file
            stats = {}
            status = "empty-or-missing"
            failed += 1

        # pick values
        siminsts = pick_first(stats, SIMINSTS_KEYS)
        ipc = pick_first(stats, IPC_KEYS)
        cpi = pick_first(stats, CPI_KEYS)
        l1d_miss = pick_first(stats, L1D_MISS_KEYS)
        l2_miss = pick_first(stats, L2_MISS_KEYS)

        # parse params from folder name
        params = parse_experiment_name(entry.name)

        row = {
            "dir": str(entry),
            "experiment": params["experiment"],
            "benchmark": params["benchmark"],
            "rob": params["rob"],
            "bp": params["bp"],
            "l1d": params["l1d"],
            "l1i": params["l1i"],
            "l2": params["l2"],
            "issue_width": params["iw"],
            "commit_width": params["cw"],
            "simInsts": siminsts,
            "IPC": ipc,
            "CPI": cpi,
            "L1D_miss_rate": l1d_miss,
            "L2_miss_rate": l2_miss,
            "console_exists": console_path.exists(),
            "status": status,
        }
        rows.append(row)

    # Write CSV
    fieldnames = [
        "dir","experiment","benchmark","rob","bp","l1d","l1i","l2","issue_width","commit_width",
        "simInsts","IPC","CPI","L1D_miss_rate","L2_miss_rate","console_exists","status"
    ]
    out_dir = Path(out_csv).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline='') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Summary print
    print(f"Scanned {total} experiment folders. Successful stats files: {succeeded}. Failed/missing: {failed}.")
    print(f"Summary written to: {out_csv}")
    # optionally show top by IPC (best-effort)
    numeric_ipc = [r for r in rows if r["IPC"] not in (None, "") and r["status"]=="ok"]
    numeric_ipc_sorted = sorted(numeric_ipc, key=lambda x: float(x["IPC"]), reverse=True)
    if numeric_ipc_sorted:
        print("\nTop 5 experiments by IPC (best available):")
        for r in numeric_ipc_sorted[:5]:
            print(f" {r['experiment']}: IPC={r['IPC']}  IPC(simInsts={r['simInsts']})  status={r['status']}")
    else:
        print("No successful IPC values found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect gem5 experiment stats into CSV")
    parser.add_argument("--experiments-dir", default="results/experiments",
                        help="Directory containing experiment subfolders")
    parser.add_argument("--output", default="results/experiments_summary.csv",
                        help="CSV output file")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    collect(args.experiments_dir, args.output, verbose=args.verbose)
