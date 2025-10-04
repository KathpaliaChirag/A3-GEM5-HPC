#!/usr/bin/env python3
import os
import shlex
from pathlib import Path

GEM5_BIN = os.environ.get("GEM5_BIN", os.path.expanduser("~/gem5/build/X86/gem5.opt"))
RUN_CUSTOM = os.environ.get("RUN_CUSTOM", os.path.expanduser("~/HPC-Projects/assignment_3/myconfigs/run_custom.py"))
OUT_ROOT = os.environ.get("OUT_ROOT", os.path.expanduser("~/HPC-Projects/assignment_3/results/experiments"))

failed_list = "failed_list.txt"
run_script = Path("rerun_failed.sh")

with open(failed_list) as f, run_script.open("w") as fh:
    fh.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
    total = 0
    for line in f:
        exp_name = line.strip()
        if not exp_name:
            continue
        outdir = os.path.join(OUT_ROOT, exp_name)
        # Reconstruct arguments from folder name
        parts = exp_name.split("_")
        bench = parts[0]
        rob = parts[1].replace("rob", "")
        bp = parts[2].replace("bp", "")
        l1_size = parts[3].replace("l1", "")
        l2_size = parts[4].replace("l2", "")
        iw = parts[5].replace("iw", "")
        cw = parts[6].replace("cw", "")
        assoc = parts[7].replace("assoc", "")
        pf = parts[8].replace("pf", "")
        rp = parts[9].replace("rp", "")

        bench_cmd = os.path.expanduser(f"~/HPC-Projects/assignment_3/benchmarks/{bench}")

        gem5_call = f"{shlex.quote(GEM5_BIN)} --outdir={shlex.quote(outdir)} {shlex.quote(RUN_CUSTOM)}"
        gem5_args = (
            f" --cmd={shlex.quote(bench_cmd)} --options=100000 "
            f"--cpu-type=DerivO3CPU --rob-size={rob} --bp-type={bp} "
            f"--l1d-size={l1_size} --l1i-size={l1_size} --l2-size={l2_size} "
            f"--l1d-assoc={assoc} --l1i-assoc={assoc} --l2-assoc={assoc} "
            f"--issue-width={iw} --commit-width={cw} --prefetcher={pf} "
            f"--repl-policy={rp} --max-insts=5000000"
        )
        full_cmd = f"{gem5_call}{gem5_args}"
        fh.write(full_cmd + "\n")
        total += 1

    fh.write(f"\necho \"All failed commands written. Total reruns: {total}\"\n")

os.chmod(run_script, 0o755)
print(f"Generated {run_script} with rerun commands. Total reruns: {total}")
