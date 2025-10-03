#!/usr/bin/env python3
import os
import itertools
import subprocess

# Paths
GEM5 = os.path.expanduser("~/gem5/build/X86/gem5.opt")
CONFIG = os.path.expanduser("~/HPC-Projects/assignment_3/myconfigs/run_custom.py")
BENCHMARK = os.path.expanduser("~/HPC-Projects/assignment_3/benchmarks/compute")

# Parameter sweeps
rob_sizes = [64, 128, 192]
issue_widths = [2, 4]
commit_widths = [2, 4]
l1d_sizes = ["32kB", "64kB"]
l2_sizes = ["256kB", "512kB"]
branch_predictors = ["LocalBP", "BiModeBP", "TournamentBP", "LTAGE"]

# New parameters
prefetchers = ["none", "stride", "tagged"]
memdeps = ["none", "simple", "storeset"]

# Max instructions
max_insts = 5_000_000

# Results directory
results_dir = "results/experiments"
os.makedirs(results_dir, exist_ok=True)

# Loop through all combinations
for rob, iw, cw, l1d, l2, bp, pf, md in itertools.product(
    rob_sizes, issue_widths, commit_widths, l1d_sizes, l2_sizes, branch_predictors, prefetchers, memdeps
):
    # Output directory
    outdir = os.path.join(
        results_dir,
        f"compute_rob{rob}_bp{bp}_l1{l1d}_l2{l2}_iw{iw}_cw{cw}_pf{pf}_md{md}"
    )
    os.makedirs(outdir, exist_ok=True)

    # Command
    cmd = [
        GEM5,
        f"--outdir={outdir}",
        CONFIG,
        f"--cmd={BENCHMARK}",
        "--options=100000",
        "--cpu-type=DerivO3CPU",
        f"--rob-size={rob}",
        f"--bp-type={bp}",
        f"--l1d-size={l1d}",
        "--l1i-size=32kB",
        f"--l2-size={l2}",
        f"--issue-width={iw}",
        f"--commit-width={cw}",
        "--l1d-assoc=2",
        "--l1i-assoc=2",
        "--l2-assoc=8",
        f"--prefetcher={pf}",
        f"--memdep={md}",
        f"--max-insts={max_insts}"
    ]

    # Run
    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Experiment failed: {outdir}")
