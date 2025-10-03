import os
import itertools
import subprocess

# Paths
GEM5_BIN = os.path.expanduser("~/gem5/build/X86/gem5.opt")
RUN_SCRIPT = os.path.expanduser("~/HPC-Projects/assignment_3/myconfigs/run_custom.py")
BENCHMARK_DIR = os.path.expanduser("~/HPC-Projects/assignment_3/benchmarks")
RESULTS_DIR = os.path.expanduser("~/HPC-Projects/assignment_3/results/experiments")

os.makedirs(RESULTS_DIR, exist_ok=True)

# Parameters
benchmarks = ["compute", "stream", "ptrchase"]
rob_sizes = [64, 128, 192]
bp_types = ["LocalBP", "BiModeBP", "TournamentBP", "LTAGE"]
l1_sizes = ["32kB", "64kB"]
l2_sizes = ["256kB", "512kB"]
issue_widths = [2, 4]
commit_widths = [2, 4]

# Fixed safe associativities
l1_assoc = 2
l2_assoc = 8

# Sweep all parameter combinations
experiments = list(itertools.product(
    benchmarks, rob_sizes, bp_types, l1_sizes, l2_sizes, issue_widths, commit_widths
))

print(f"Total experiments: {len(experiments)}")

for bench, rob, bp, l1, l2, iw, cw in experiments:
    outdir = os.path.join(
        RESULTS_DIR,
        f"{bench}_rob{rob}_bp{bp}_l1{l1}_l2{l2}_iw{iw}_cw{cw}"
    )
    os.makedirs(outdir, exist_ok=True)

    cmd = [
        GEM5_BIN,
        f"--outdir={outdir}",
        RUN_SCRIPT,
        f"--cmd={os.path.join(BENCHMARK_DIR, bench)}",
        "--options=100000",  # workload argument
        "--cpu-type=DerivO3CPU",
        f"--rob-size={rob}",
        f"--bp-type={bp}",
        f"--l1d-size={l1}",
        f"--l1i-size={l1}",   # keep L1I = L1D
        f"--l2-size={l2}",
        f"--l1d-assoc={l1_assoc}",
        f"--l1i-assoc={l1_assoc}",
        f"--l2-assoc={l2_assoc}",
        f"--issue-width={iw}",
        f"--commit-width={cw}",
        "--max-insts=5000000"
    ]

    print(f"\nRunning: {outdir}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print(f"Experiment failed: {outdir}")
