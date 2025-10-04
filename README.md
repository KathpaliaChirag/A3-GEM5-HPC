# Assignment 3: Out-of-Order CPU Design Exploration using gem5

Author: CK  
Institute: IIT Delhi, M.Tech CSE  
Date: October 2025  
Project: HPC Assignment 3  
Simulator: gem5 v23.1.0.0 (X86 target)

=====================================================================
1. gem5 Setup and Environment
=====================================================================

Installed gem5 version 23.1.0.0 from source and built successfully:

    scons build/X86/gem5.opt -j$(nproc)

Verified the build:

    ./build/X86/gem5.opt --version

Output confirmed version 23.1.0.0, compiler details, and build success.

Project folder initialized:

    mkdir -p ~/HPC-Projects/assignment_3

Directory structure after setup:

    HPC-Projects/assignment_3/
      ├── benchmarks/
      ├── scripts/
      ├── myconfigs/
      ├── results/
      ├── Makefile
      ├── README.txt

=====================================================================
2. Benchmarks Implemented
=====================================================================

We implemented three benchmarks to cover different CPU and memory behaviors.

---------------------------------------------------------------------
2A. compute.c
---------------------------------------------------------------------
- Type: CPU-bound workload
- Description: Performs repeated floating-point operations (arithmetic loop)
- Goal: Stress core pipeline (ALU, FPU)
- Metrics: IPC, CPI
- Expected: High IPC, low CPI (compute-intensive)
- Command:

      gcc compute.c -O2 -o compute

Run test:

      ./compute 100000

---------------------------------------------------------------------
2B. ptrchase.c
---------------------------------------------------------------------
- Type: Memory latency test
- Description: Random pointer chasing through linked nodes
- Goal: Stress memory latency and cache miss penalty
- Metrics: L1D miss rate, CPI
- Expected: Low IPC due to frequent cache misses
- Command:

      gcc ptrchase.c -O2 -o ptrchase

Run test:

      ./ptrchase 100000

---------------------------------------------------------------------
2C. stream.c
---------------------------------------------------------------------
- Type: Memory bandwidth test
- Description: Implements STREAM triad (C = A + α * B)
- Goal: Stress sequential memory throughput
- Metrics: Bandwidth, IPC
- Expected: Moderate IPC, memory bandwidth limited
- Command:

      gcc stream.c -O2 -o stream

Run test:

      ./stream 100000

=====================================================================
3. Makefile for Benchmarks
=====================================================================

Created a Makefile to build all benchmarks with a single command:

    make all

This builds compute, ptrchase, and stream binaries.

=====================================================================
4. Baseline Host Execution
=====================================================================

All benchmarks were verified on host (Ubuntu):

    ./compute 100000000
    ./ptrchase 100000000
    ./stream 100000000

Results:

    compute -> done 314159031.099287
    ptrchase -> end 30493232
    stream -> done 7.283180

These form baseline runtime and correctness validation before gem5 simulation.

=====================================================================
5. Initial gem5 Simulation
=====================================================================

Used example config for basic simulation:

    ~/gem5/build/X86/gem5.opt configs/deprecated/example/se.py \
      -c ./benchmarks/compute -o "100000" \
      --cpu-type=DerivO3CPU --caches --l2cache \
      --l1d_size=32kB --l1i_size=32kB --l2_size=256kB \
      -I 50000000

Verified simulation completed successfully, producing stats.txt.

=====================================================================
6. Extracting Stats
=====================================================================

Implemented scripts/parse_stats.py to extract:
- simInsts
- IPC
- CPI
- L1D miss rate
- L2 miss rate

Usage:

    cd scripts
    python3 parse_stats.py ../results/run*/stats.txt

=====================================================================
7. Custom Config: run_custom.py
=====================================================================

Developed a modular config file (myconfigs/run_custom.py) with CLI arguments:

    --cmd, --options
    --rob-size
    --issue-width
    --commit-width
    --bp-type
    --l1d-size, --l1i-size, --l2-size
    --prefetcher {none,stride,tagged}
    --repl-policy {LRU,Random}
    --max-insts

Example run:

    ~/gem5/build/X86/gem5.opt myconfigs/run_custom.py \
      --cmd /bin/ls --options="-la" \
      --prefetcher none --memdep none --max-insts 10000

This became our main experiment driver configuration.

=====================================================================
8. Prefetcher and MemDep Predictor Options
=====================================================================

Prefetchers supported:
    --prefetcher none
    --prefetcher stride
    --prefetcher tagged

Memory Dependence (optional bonus):
    --memdep none
    --memdep simple
    --memdep storeset

The memdep predictor is not supported in gem5’s X86 DerivO3CPU;
documented as optional (not attempted).

=====================================================================
9. Experiment Automation: run_experiments.py
=====================================================================

Created scripts/run_experiments.py to auto-generate all experiment runs.

Parameter sweeps included:

    Benchmarks: compute, ptrchase, stream
    ROB sizes: 64, 128, 192
    Issue widths: 2, 4
    Commit widths: 2, 4
    L1 sizes: 32kB, 64kB
    L2 sizes: 256kB, 512kB
    Associativities: 2, 4
    Branch predictors: none, LocalBP, BiModeBP, TournamentBP
    Prefetchers: none, stride, tagged
    Replacement policies: LRU, Random

Total = 3 * 3 * 2 * 2 * 2 * 2 * 2 * 4 * 3 * 2 = 6912 runs.

Generated:

    run_all.sh

To run sequentially:

    ./run_all.sh

=====================================================================
10. Parallel Execution
=====================================================================

To utilize all 16 threads of Ryzen 7 5800H:

    parallel -j8 < run_all.sh

This executes 8 gem5 runs in parallel, achieving full CPU utilization.

Checked CPU cores:

    nproc
    lscpu | grep -E 'Model name|Socket|Core|Thread'

=====================================================================
11. Monitoring and Recovery
=====================================================================

To check running jobs:

    pgrep -af gem5 | wc -l

To collect results mid-run:

    python3 scripts/collect_results.py

This script scanned all experiments, categorized into:
- ok
- empty-or-missing

Result file: results/experiments_summary.csv

Example output:

    Scanned 6912 experiment folders.
    Successful stats files: 4415
    Failed/missing: 2497

=====================================================================
12. Debug and Error Tracking
=====================================================================

Common errors and fixes:

- ImportError: no module named configs  
  → fixed by direct m5.objects imports.

- ValueError: cannot convert 'l132kB'  
  → removed typo 'l132kB' (should be '32kB').

- Assertion failure: TimeBuffer<T>::valid  
  → caused by aggressive speculation (large ROB, stride pf).

- AttributeError: Class X86O3CPU has no parameter memDepPred  
  → removed memdep parameter since unsupported.

=====================================================================
13. Failure Analysis
=====================================================================

Post-collection failure summary:

    grep "empty-or-missing" results/experiments_summary.csv | cut -d, -f2 | wc -l
    → 2497 failures out of 6912

Benchmark-wise:

    768 compute
    769 ptrchase
    960 stream

Branch predictor-wise:

    656 BiModeBP
    593 LocalBP
    624 TournamentBP
    624 none

Prefetcher-wise:

    1410 pfnone
    1920 pfstride
    1664 pftagged

ROB size-wise:

    1152 rob64
    1856 rob128
    1986 rob192

Observation:
Stride prefetcher and large ROB sizes most unstable.

=====================================================================
14. Rerun Mechanism
=====================================================================

To rerun failed cases only:

    grep "empty-or-missing" results/experiments_summary.csv | cut -d, -f2 > failed_list.txt
    python3 scripts/rerun_failed.py

This generated rerun_failed.sh with only failed experiment commands.

To execute:

    parallel -j8 < rerun_failed.sh

=====================================================================
15. Post-Rerun Summary
=====================================================================

After reruns:

    python3 scripts/collect_results.py

Example:

    Scanned 6912 experiment folders.
    Successful stats: 4415
    Failed/missing: 2497

Used AWK and grep to classify failure causes.

=====================================================================
16. Data Extraction and Classification
=====================================================================

Commands used:

    awk -F, 'NR>1{gsub(/^[ \t]+|[ \t\r]+$/, "", $NF); counts[$NF]++} END{for (k in counts) print k, counts[k]}' results/experiments_summary.csv

    grep "empty-or-missing" results/experiments_summary.csv | cut -d, -f2 | cut -d_ -f1 | sort | uniq -c
    grep "empty-or-missing" results/experiments_summary.csv | grep -o "pf[^_]*" | sort | uniq -c
    grep "empty-or-missing" results/experiments_summary.csv | grep -o "rob[0-9]*" | sort | uniq -c

This produced structured classification by benchmark, predictor, prefetcher, and ROB size.

=====================================================================
17. Observations and Trends
=====================================================================

- Increasing ROB and issue width improved IPC until gem5 instability appeared.
- Prefetchers improved IPC, but stride prefetcher caused most assertion failures.
- TournamentBP provided highest IPC but marginal improvement over BiModeBP.
- Replacement policy (LRU vs Random) had minimal IPC impact.
- Branch speculation always enabled (no explicit disable mode implemented).
- Memory-dependence speculation: not attempted (unsupported in X86 build).

=====================================================================
18. Top Performing Configurations
=====================================================================

compute_rob128_bpTournamentBP_l164kB_l2512kB_iw4_cw4_assoc4_pftagged_rpLRU
IPC = 1.3643

compute_rob64_bpTournamentBP_l164kB_l2512kB_iw4_cw4_assoc4_pftagged_rpLRU
IPC = 1.3642

compute_rob64_bpnone_l164kB_l2512kB_iw4_cw4_assoc4_pftagged_rpLRU
IPC = 1.3642

=====================================================================
19. Cleanup and Maintenance
=====================================================================

To remove all results before reruns:

    rm -rf results/experiments/*

To clean selectively (e.g., rob64 and rob128):

    rm -rf results/experiments/*rob64* results/experiments/*rob128*

=====================================================================
20. Summary of Work Completed
=====================================================================

Implemented:
- 3 benchmarks (compute, ptrchase, stream)
- 3 cache levels (L1I, L1D, L2)
- 4 branch predictors (none, Local, BiMode, Tournament)
- 3 prefetchers (none, stride, tagged)
- 2 replacement policies (LRU, Random)
- 3 ROB sizes (64, 128, 192)
- Issue/Commit width sweeps (2, 4)
- Full automation via Python scripts
- Failure recovery and classification
- Parallel execution optimization (8-core batch)
- Comprehensive result aggregation and IPC ranking

Total runs: 6912  
Successful: 4415  
Failed: 2497  
Failure rate: ~36%

=====================================================================
21. Future Improvements
=====================================================================

- Add branch speculation disable mode (--no-bpred).
- Add replacement policy variations beyond LRU/Random.
- Attempt StoreSet memory dependence predictor (requires C++ rebuild).
- Automate graph generation for IPC, CPI, miss rates via matplotlib.
- Implement checkpoint-based resume for long simulations.

=====================================================================
22. Key Scripts and Their Roles
=====================================================================

myconfigs/run_custom.py
    Core config file for gem5 runs.

scripts/run_experiments.py
    Generates run_all.sh for all parameter combinations.

scripts/collect_results.py
    Parses all stats.txt files and aggregates into summary CSV.

scripts/rerun_failed.py
    Regenerates rerun_failed.sh for missing/failed runs.

scripts/parse_stats.py
    Extracts key performance metrics (IPC, CPI, miss rates).

=====================================================================
23. Conclusion
=====================================================================

This project successfully explored Out-of-Order CPU performance using gem5 by systematically varying microarchitectural parameters and analyzing their effects on IPC and stability.

We achieved a modular, automated workflow for:
- Benchmark simulation
- Parameter sweeping
- Prefetcher and speculation evaluation
- Automated failure recovery and analysis

The framework can be reused for further exploration of memory systems, speculation, and replacement policies.


### ADD-ON : Details of failed runs :
chira@chirag:~/HPC-Projects/assignment_3$ grep "empty-or-missing" results/experiments_summary.csv | cut -d, -f2 | cut -d_ -f1 | sort | uniq -c
    768 compute
    769 ptrchase
    960 stream
chira@chirag:~/HPC-Projects/assignment_3$ grep "empty-or-missing" results/experiments_summary.csv \
| cut -d, -f2 \
| grep -o "bp[^_]*" \
| sort | uniq -c
    656 bpBiModeBP
    593 bpLocalBP
    624 bpTournamentBP
    624 bpnone
chira@chirag:~/HPC-Projects/assignment_3$ grep "empty-or-missing" results/experiments_summary.csv \
| grep -o "pf[^_]*" \
| sort | uniq -c
   1410 pfnone
   1920 pfstride
   1664 pftagged
chira@chirag:~/HPC-Projects/assignment_3$ grep "empty-or-missing" results/experiments_summary.csv \
| grep -o "rob[0-9]*" \
| sort | uniq -c
   1856 rob128
   1986 rob192
   1152 rob64
chira@chirag:~/HPC-Projects/assignment_3$ grep "empty-or-missing" results/experiments_summary.csv \
| awk -F, '{split($2,a,"_"); print a[1],a[3],a[9]}' \
| sort | uniq -c | column -t
48  compute   bpBiModeBP      pfnone
80  compute   bpBiModeBP      pfstride
80  compute   bpBiModeBP      pftagged
48  compute   bpLocalBP       pfnone
80  compute   bpLocalBP       pfstride
48  compute   bpLocalBP       pftagged
48  compute   bpTournamentBP  pfnone
80  compute   bpTournamentBP  pfstride
64  compute   bpTournamentBP  pftagged
48  compute   bpnone          pfnone
80  compute   bpnone          pfstride
64  compute   bpnone          pftagged
48  ptrchase  bpBiModeBP      pfnone
80  ptrchase  bpBiModeBP      pfstride
80  ptrchase  bpBiModeBP      pftagged
49  ptrchase  bpLocalBP       pfnone
80  ptrchase  bpLocalBP       pfstride
48  ptrchase  bpLocalBP       pftagged
48  ptrchase  bpTournamentBP  pfnone
80  ptrchase  bpTournamentBP  pfstride
64  ptrchase  bpTournamentBP  pftagged
48  ptrchase  bpnone          pfnone
80  ptrchase  bpnone          pfstride
64  ptrchase  bpnone          pftagged
80  stream    bpBiModeBP      pfnone
80  stream    bpBiModeBP      pfstride
80  stream    bpBiModeBP      pftagged
80  stream    bpLocalBP       pfnone
80  stream    bpLocalBP       pfstride
80  stream    bpLocalBP       pftagged
80  stream    bpTournamentBP  pfnone
80  stream    bpTournamentBP  pfstride
80  stream    bpTournamentBP  pftagged
80  stream    bpnone          pfnone
80  stream    bpnone          pfstride
80  stream    bpnone          pftagged
chira@chirag:~/HPC-Projects/assignment_3$
