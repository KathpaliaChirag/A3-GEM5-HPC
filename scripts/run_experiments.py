#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path

# Paths
gem5 = str(Path.home() / "gem5/build/X86/gem5.opt")
script = str(Path.cwd().parent / "configs/run_custom.py")
benchmarks = {
    "compute": "../benchmarks/compute 100000",
    "ptrchase": "../benchmarks/ptrchase 100000",
    "stream": "../benchmarks/stream 100000"
}

# Experiment parameters
rob_sizes = [64, 128, 192]
bp_types = ["TournamentBP", "LocalBP", "BiModeBP"]
l1_sizes = ["32kB", "64kB"]
l2_sizes = ["256kB", "512kB"]

# Outdir base
results_dir = Path("../results/experiments")
results_dir.mkdir(parents=True, exist_ok=True)

# Run sweeps
for bench, cmd in benchmarks.items():
    for rob in rob_sizes:
        for bp in bp_types:
            for l1 in l1_sizes:
                for l2 in l2_sizes:
                    outdir = results_dir / f"{bench}_rob{rob}_bp{bp}_l1{l1}_l2{l2}"
                    outdir.mkdir(parents=True, exist_ok=True)

                    run_cmd = [
                        gem5,
                        f"--outdir={outdir}",
                        script,
                        f"--cmd={cmd.split()[0]}",
                        f"--options={' '.join(cmd.split()[1:])}",
                        "--cpu-type=DerivO3CPU",
                        f"--rob-size={rob}",
                        f"--bp-type={bp}",
                        f"--l1d-size={l1}",
                        f"--l1i-size={l1}",
                        f"--l2-size={l2}",
                        "--issue-width=4",
                        "--commit-width=4",
                        "--max-insts=5000000"
                    ]

                    print("Running:", " ".join(run_cmd))
                    try:
                        subprocess.run(run_cmd, check=True)
                    except subprocess.CalledProcessError:
                        print(f"[ERROR] gem5 failed for {outdir}, check console.txt")

