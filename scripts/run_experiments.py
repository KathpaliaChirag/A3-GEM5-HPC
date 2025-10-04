#!/usr/bin/env python3
"""
Generate a batch of gem5 experiment commands.

Now generates ~4608 runs:
  benchmarks (3) x robs (2) x iw (2) x cw (2) x l1 (2) x l2 (2) x assoc (2)
  x bp (4: none,Local,BiMode,Tournament) x prefetchers (3) x repl (2)
= 4608 runs

By default this script writes run_all.sh containing all gem5 commands.
Set EXECUTE = True to run sequentially (be careful).
"""

import os
import shlex
from pathlib import Path
from os.path import expanduser

# User-editable: path to gem5 binary and run_custom path
GEM5_BIN = os.environ.get("GEM5_BIN", expanduser("~/gem5/build/X86/gem5.opt"))
RUN_CUSTOM = os.environ.get("RUN_CUSTOM", expanduser("~/HPC-Projects/assignment_3/myconfigs/run_custom.py"))
OUT_ROOT = os.environ.get("OUT_ROOT", expanduser("~/HPC-Projects/assignment_3/results/experiments"))
BENCH_BASE = expanduser("~/HPC-Projects/assignment_3/benchmarks")

# Parameters to sweep (tuned to produce ~4608 runs)
benchmarks = [
    ("compute", "100000"),
    ("ptrchase", "100000"),
    ("stream", "100000"),
]

#robs = [192]              # chosen to yield ~4608 total runs
#issue_widths = [2, 4]
#commit_widths = [2, 4]
#l1_sizes = ["32kB", "64kB"]
#l2_sizes = ["256kB", "512kB"]
#assocs = [2, 4]
#bps = ["none", "LocalBP", "BiModeBP", "TournamentBP"]
#prefetchers = ["none", "stride", "tagged"]
#repl_policies = ["LRU", "Random"]

robs = [64, 128, 192]
issue_widths = [2, 4]
commit_widths = [2, 4]
l1_sizes = ["32kB", "64kB"]
l2_sizes = ["256kB", "512kB"]
assocs = [2, 4]
bps = ["none", "LocalBP", "BiModeBP", "TournamentBP"]
prefetchers = ["none", "stride", "tagged"]
repl_policies = ["LRU", "Random"]


# Execution control
EXECUTE = False   # If True, run sequentially (be careful). Default: just generate run_all.sh
DRY_RUN = False   # If True, only print a quick summary, don't write run_all.sh

Path("results").mkdir(parents=True, exist_ok=True)
Path(OUT_ROOT).mkdir(parents=True, exist_ok=True)

# Output script
run_script = Path("run_all.sh")
with run_script.open("w") as fh:
    fh.write("#!/usr/bin/env bash\n")
    fh.write("set -euo pipefail\n\n")

    total = 0
    for bench_name, opts in benchmarks:
        bench_cmd = os.path.join(BENCH_BASE, bench_name)
        for rob in robs:
            for iw in issue_widths:
                for cw in commit_widths:
                    for l1_size in l1_sizes:
                        for l2_size in l2_sizes:
                            for assoc in assocs:
                                for bp in bps:
                                    for pf in prefetchers:
                                        for rp in repl_policies:
                                            # Folder naming: include l1/l2 labels for readability
                                            l1_label = f"l1{l1_size}"
                                            l2_label = f"l2{l2_size}"

                                            outdir_name = (
                                                f"{bench_name}_rob{rob}_bp{bp}_{l1_label}_{l2_label}"
                                                f"_iw{iw}_cw{cw}_assoc{assoc}_pf{pf}_rp{rp}"
                                            )
                                            outdir = os.path.join(OUT_ROOT, outdir_name)
                                            os.makedirs(outdir, exist_ok=True)

                                            # Build gem5 command
                                            gem5_call = f"{shlex.quote(GEM5_BIN)} --outdir={shlex.quote(outdir)} {shlex.quote(RUN_CUSTOM)}"
                                            gem5_args = (
                                                f" --cmd={shlex.quote(bench_cmd)} --options={shlex.quote(opts)} "
                                                f"--cpu-type=DerivO3CPU --rob-size={rob} --bp-type={bp} "
                                                f"--l1d-size={l1_size} --l1i-size={l1_size} --l2-size={l2_size} "
                                                f"--l1d-assoc={assoc} --l1i-assoc={assoc} --l2-assoc={assoc} "
                                                f"--issue-width={iw} --commit-width={cw} --prefetcher={pf} "
                                                f"--repl-policy={rp} --max-insts=5000000"
                                            )

                                            full_cmd = f"{gem5_call}{gem5_args}"
                                            fh.write(full_cmd + "\n")
                                            total += 1

    fh.write(f"\necho \"All commands written. Total runs: {total}\"\n")

# make the script executable
os.chmod(run_script, 0o755)

print(f"Generated {run_script} with all commands. Total runs: {total}")
print("Inspect the file before executing. To run sequentially use: ./run_all.sh")
if EXECUTE:
    print("EXECUTE=True: running commands sequentially (this may take a long time)...")
    os.system(str(run_script))
