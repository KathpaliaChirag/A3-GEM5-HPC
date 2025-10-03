#!/usr/bin/env python3
import os
import subprocess

summary_file = "results/experiments_summary.csv"
gem5_bin = os.path.expanduser("~/gem5/build/X86/gem5.opt")
run_script = os.path.expanduser("~/HPC-Projects/assignment_3/myconfigs/run_custom.py")
benchmark = os.path.expanduser("~/HPC-Projects/assignment_3/benchmarks/compute")

def parse_config(name):
    """Parse experiment name like compute_rob128_bpBiModeBP_l132kB_l2256kB into params."""
    parts = name.split("_")
    config = {
        "bench": parts[0],
        "rob": parts[1].replace("rob", ""),
        "bp": parts[2].replace("bp", ""),
        "l1d": parts[3].replace("l1", "").lstrip("l"),
        "l2": parts[4].replace("l2", "").lstrip("l"),
        "iw": 2,
        "cw": 2,
    }
    if "iw4" in name:
        config["iw"] = 4
    if "cw4" in name:
        config["cw"] = 4
    return config

def run_experiment(name, config):
    outdir = os.path.join("results/experiments", name)
    cmd = [
        gem5_bin,
        f"--outdir={outdir}",
        run_script,
        f"--cmd={benchmark}",
        "--options=100000",
        "--cpu-type=DerivO3CPU",
        f"--rob-size={config['rob']}",
        f"--bp-type={config['bp']}",
        f"--l1d-size={config['l1d']}",
        "--l1i-size=32kB",
        f"--l2-size={config['l2']}",
        f"--issue-width={config['iw']}",
        f"--commit-width={config['cw']}",
        "--l1d-assoc=2",
        "--l1i-assoc=2",
        "--l2-assoc=8",
        "--max-insts=5000000",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd)

def main():
    with open(summary_file) as f:
        for line in f:
            if "empty-or-missing" in line:
                name = line.split(",")[1].strip()
                config = parse_config(name)

                # (optional) skip LTAGE if you don’t want it
                if config["bp"] == "LTAGE":
                    continue

                run_experiment(name, config)

if __name__ == "__main__":
    main()
