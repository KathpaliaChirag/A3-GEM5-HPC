# HPC Project 3

## Step 1: Setup
- Installed fresh gem5 (v23.1).
- Successfully built gem5 with scons (X86 target).
- Project folder assignment_3 initialized.
## Step 2A: Benchmarks (added compute)

- Added `benchmarks/compute.c`.
- Purpose: CPU-intensive workload that performs floating-point arithmetic in a loop.
- Stresses processor core (ALU, FPU) rather than memory.
- Will help us measure IPC performance of DerivO3CPU in gem5 under compute-heavy conditions.
 
## Step 2B: Benchmarks (added Pointer chase)


- Added `benchmarks/ptrchase.c`.
- Purpose: memory latency stress test using random pointer chasing.
- Accesses memory unpredictably → causes cache misses.
- Useful to measure L1D miss rate and IPC under memory-bound conditions.

Allocates an array of “nodes” (each roughly a cache line).
Each node points to another node → forms a linked list.
The list is randomized (so memory accesses are unpredictable).
Then it repeatedly follows pointers for 1,000,000 iterations.
This creates random memory access patterns.

Why it matters for the assignment
This benchmark stresses memory latency rather than compute.
Modern CPUs are fast, but chasing random pointers causes cache misses → CPU stalls.
Running this in gem5 lets you measure:
L1D cache miss rate
Impact of memory hierarchy on IPC
Comparing this with compute.c highlights how memory bottlenecks hurt performance.

## Step 2C: Benchmarks (added Stream)

- Added `benchmarks/stream.c`.
- Purpose: memory bandwidth stress test using the STREAM triad (C = A + α*B).
- Sequential array accesses that saturate memory bandwidth.
- Useful to evaluate memory throughput and its impact on IPC.

Allocates 3 large arrays A, B, C.

Initializes them with some values.

Performs the classic STREAM triad:

			C[i]=A[i]+α×B[i]

Loops multiple times to amplify runtime.
Prints the last element to prevent compiler from optimizing the loop away.

📌 Why it matters
This benchmark is memory-bandwidth intensive.
It stresses throughput of memory subsystem (L1 → L2 → DRAM).
Unlike ptrchase (random access), this is sequential streaming access.
Running it in gem5 lets you measure:
How much bandwidth your memory system can sustain.
The difference between compute-bound, latency-bound, and bandwidth-bound workloads.


## Step 3: Makefile added for easier computation

- Added Makefile in benchmarks folder.
- Allows compiling all benchmarks (`compute`, `ptrchase`, `stream`) with a single `make` command.
- Verified by running `./compute 1000` successfully.


## Step 4: Run Benchmarks Natively (Baseline)

- Ran all benchmarks on host (Ubuntu).
- Verified that they run to completion and print results.
- This provides a baseline runtime for comparison with gem5 simulations.

Commands used : 
- ./compute 100000000
- ./ptrchase 100000000
- ./stream 100000000

- Results:
  - compute 100000000 → done 314159031.099287
  - ptrchase 100000000 → end 30493232
  - stream 100000000 → done 7.283180
- Verified correctness and runtime — provides baseline for gem5 simulation.
Before using gem5, we always run the benchmarks on the host system (your WSL/Ubuntu). This gives you:

- A baseline runtime.
- Proof that the benchmarks actually work.
- Something to compare with gem5 simulation results.


### Step 5: First gem5 Run
- Ran `compute` benchmark inside gem5 using **DerivO3CPU** with L1/L2 caches.
- Command used:
  ```bash
  ~/gem5/build/X86/gem5.opt ~/gem5/configs/deprecated/example/se.py \
    -c ./benchmarks/compute -o "100000" \
    --cpu-type=DerivO3CPU --caches --l2cache \
    --l1d_size=32kB --l1i_size=32kB --l2_size=256kB \
    -I 50000000
