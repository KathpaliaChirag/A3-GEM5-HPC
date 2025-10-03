import os
import subprocess

gem5_bin = os.path.expanduser("~/gem5/build/X86/gem5.opt")
config_script = "configs/run_custom.py"
benchmark_bin = "./benchmarks/compute"
options = "100000"

# Read failed experiments list
with open("failed_experiments.txt") as f:
    failed = [line.strip() for line in f if line.strip()]

for exp in failed:
    outdir = os.path.join("results/experiments", exp)
    print(f"Rerunning: {exp}")

    cmd = [
        gem5_bin,
        f"--outdir={outdir}",
        config_script,
        "--cmd", benchmark_bin,
        "--options", options,
        "--cpu-type=DerivO3CPU",
        "--rob-size", exp.split("_")[1].replace("rob",""),
        "--bp-type", exp.split("_")[2].replace("bp",""),
        "--l1d-size", exp.split("_")[3].replace("l1d","").replace("l",""),
        "--l1i-size", "32kB",   # fallback if missing
        "--l2-size", exp.split("_")[4].replace("l2",""),
        "--issue-width", "2",   # default
        "--commit-width", "2",  # default
        "--max-insts", "5000000"
    ]

    subprocess.run(cmd)
