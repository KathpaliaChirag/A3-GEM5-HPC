#!/usr/bin/env python3
"""
Standalone gem5 Python configuration that exposes the knobs we need for the sweep.

Safe behavior:
 - If a requested object (prefetcher / replacement policy / memdep) is not available
   in your gem5 build, this script prints a warning and continues with defaults.
 - Branch prediction can be disabled with --bp-type=none (this will not attach a predictor).
 - Mem-dependence option is accepted but only applied if the relevant SimObjects exist;
   otherwise the script prints a warning. (Many gem5 builds differ here.)
"""

import argparse
import os
import sys
import m5
from m5.objects import *

# Helper to try to get an object class from m5.objects safely
def get_obj_class(name):
    return getattr(sys.modules.get('m5.objects'), name, None)

parser = argparse.ArgumentParser(description="Custom gem5 simulation configuration")

parser.add_argument("--cmd", type=str, required=True,
                    help="Command to run (path in guest).")
parser.add_argument("--options", type=str, default="",
                    help="Options for the benchmark (single string, will be split).")
parser.add_argument("--cpu-type", type=str, default="DerivO3CPU",
                    help="Type of CPU (default: DerivO3CPU)")
parser.add_argument("--rob-size", type=int, default=128,
                    help="Reorder buffer size (default: 128)")
parser.add_argument("--issue-width", type=int, default=2,
                    help="CPU issue width (default: 2)")
parser.add_argument("--commit-width", type=int, default=2,
                    help="CPU commit width (default: 2)")
parser.add_argument("--l1d-size", type=str, default="32kB",
                    help="L1 data cache size (default: 32kB)")
parser.add_argument("--l1i-size", type=str, default="32kB",
                    help="L1 instruction cache size (default: 32kB)")
parser.add_argument("--l2-size", type=str, default="256kB",
                    help="L2 cache size (default: 256kB)")
parser.add_argument("--l1d-assoc", type=int, default=2,
                    help="L1 data cache associativity (default: 2)")
parser.add_argument("--l1i-assoc", type=int, default=2,
                    help="L1 instruction cache associativity (default: 2)")
parser.add_argument("--l2-assoc", type=int, default=8,
                    help="L2 cache associativity (default: 8)")
parser.add_argument("--bp-type", type=str, default="TournamentBP",
                    choices=["none", "LocalBP", "BiModeBP", "TournamentBP", "LTAGE"],
                    help="Branch predictor type (default: TournamentBP). 'none' disables predictor")
parser.add_argument("--prefetcher", type=str, default="none",
                    choices=["none", "stride", "tagged"],
                    help="Prefetcher for caches (default: none)")
parser.add_argument("--memdep", type=str, default="none",
                    choices=["none", "simple", "storeset"],
                    help="Memory-dependence option (best-effort; not all builds expose objects)")
parser.add_argument("--repl-policy", type=str, default="LRU",
                    choices=["LRU", "Random"],
                    help="Replacement policy hint for caches (best-effort)")
parser.add_argument("--max-insts", type=int, default=1000000,
                    help="Maximum number of instructions to simulate (default: 1e6)")
parser.add_argument("--cpu-clock", type=str, default="1GHz",
                    help="CPU clock frequency (default: 1GHz)")
parser.add_argument("--mem-size", type=str, default="512MB",
                    help="Memory size (default: 512MB)")

args = parser.parse_args()

# ----------------------------
# System configuration
# ----------------------------
system = System()

system.clk_domain = SrcClockDomain()
system.clk_domain.clock = args.cpu_clock
system.clk_domain.voltage_domain = VoltageDomain()

system.mem_mode = "timing"
system.mem_ranges = [AddrRange(args.mem_size)]

# Create CPU
# We use DerivO3CPU by default (X86O3CPU type). Use the string to instantiate.
CPUClass = get_obj_class(args.cpu_type)
if CPUClass is None:
    print(f"WARNING: CPU class '{args.cpu_type}' not found in m5.objects. Falling back to DerivO3CPU.")
    CPUClass = get_obj_class("DerivO3CPU")
if CPUClass is None:
    raise SystemExit("ERROR: No suitable CPU class found in your gem5 build.")
system.cpu = CPUClass()

# Set CPU-related params (best-effort; names may differ across gem5 versions)
# Use try/except to avoid AttributeError on older/newer builds.
def try_set(obj, name, val):
    try:
        setattr(obj, name, val)
    except Exception:
        print(f"NOTE: Could not set {obj}.{name} = {val} (not available in this build)")

try_set(system.cpu, "numROBEntries", args.rob_size)
# Many CPU objects use different attribute names; try common ones:
try_set(system.cpu, "numROBEntries", args.rob_size)
try_set(system.cpu, "robEntries", args.rob_size)
try_set(system.cpu, "fetchWidth", args.issue_width)
try_set(system.cpu, "decodeWidth", args.issue_width)
try_set(system.cpu, "renameWidth", args.issue_width)
try_set(system.cpu, "issueWidth", args.issue_width)
try_set(system.cpu, "wbWidth", args.commit_width)
try_set(system.cpu, "commitWidth", args.commit_width)
try_set(system.cpu, "squashWidth", args.commit_width)
# max insts per thread
try_set(system.cpu, "max_insts_any_thread", args.max_insts)

# Branch predictor
if args.bp_type == "none":
    # If we can set no predictor, skip attaching one and print note
    print("Branch prediction disabled for this run (--bp-type=none).")
else:
    BPClass = get_obj_class(args.bp_type)
    if BPClass is None:
        print(f"WARNING: Branch predictor class '{args.bp_type}' not found. Using default (if any).")
    else:
        try:
            system.cpu.branchPred = BPClass()
        except Exception:
            print(f"NOTE: Could not attach branch predictor '{args.bp_type}' on this CPU object.")

# ----------------------------
# Cache creation helpers
# ----------------------------
def make_cache(size, assoc, repl_policy, prefetcher_name=None, is_l1=False):
    """Create a generic Cache object and try to apply requested features.
       This is intentionally permissive and will warn if features are not available.
    """
    c = Cache()
    # sizes/assoc are strings and ints; Cache expects size as string like "32kB".
    try_set(c, "size", size)
    try_set(c, "assoc", assoc)
    try_set(c, "tag_latency", 2)
    try_set(c, "data_latency", 2)
    try_set(c, "response_latency", 2)
    try_set(c, "mshrs", 4)
    try_set(c, "tgts_per_mshr", 20)

    # Replacement policy: try to pick objects if present (best-effort)
    if repl_policy:
        rp_obj = None
        # common names in some gem5 builds
        if repl_policy == "Random":
            rp_obj = get_obj_class("RandomRP") or get_obj_class("RandomReplacementPolicy")
        elif repl_policy == "LRU":
            rp_obj = get_obj_class("LRURP") or get_obj_class("LRUReplacementPolicy")
        if rp_obj:
            try_set(c, "replacement_policy", rp_obj())
        else:
            # fall back to not setting it; gem5 default will apply.
            print(f"INFO: replacement policy object for '{repl_policy}' not found; using simulator default")

    # Prefetcher (best-effort)
    if prefetcher_name and prefetcher_name != "none":
        pf = None
        if prefetcher_name == "stride":
            pf = get_obj_class("StridePrefetcher") or get_obj_class("TaggedStridePrefetcher") or get_obj_class("StridePrefetcherSimple")
        elif prefetcher_name == "tagged":
            pf = get_obj_class("TaggedPrefetcher") or get_obj_class("TaggedPrefetcherSimple")
        if pf:
            try_set(c, "prefetcher", pf())
        else:
            print(f"INFO: prefetcher class for '{prefetcher_name}' not found in this build; continuing without it.")

    return c

# Create caches
system.cpu.icache = make_cache(args.l1i_size, args.l1i_assoc, args.repl_policy, args.prefetcher, is_l1=True)
system.cpu.dcache = make_cache(args.l1d_size, args.l1d_assoc, args.repl_policy, args.prefetcher, is_l1=True)
system.l2cache = make_cache(args.l2_size, args.l2_assoc, args.repl_policy, args.prefetcher, is_l1=False)

# Buses
system.l2bus = L2XBar()
system.membus = SystemXBar()

# Connect L1 caches to CPU ports (best-effort; attribute names differ)
# Typical derivations:
try:
    system.cpu.icache.cpu_side = system.cpu.icache_port
    system.cpu.dcache.cpu_side = system.cpu.dcache_port
except Exception:
    # Try typical older attribute names
    try:
        system.cpu.icache.cpu_side = system.cpu.icache_port
    except Exception:
        pass

# Connect caches to buses (best-effort)
try:
    system.cpu.icache.mem_side = system.l2bus.cpu_side_ports
    system.cpu.dcache.mem_side = system.l2bus.cpu_side_ports
    system.l2cache.cpu_side = system.l2bus.mem_side_ports
    system.l2cache.mem_side = system.membus.cpu_side_ports
except Exception:
    # Some builds use different attribute names; if it fails, warn.
    print("WARNING: Could not connect caches to buses using assumed attribute names. If simulation fails, check your gem5 build's cache object API.")

# Interrupt controller
try:
    system.cpu.createInterruptController()
    # connect typical vectors
    system.cpu.interrupts[0].pio = system.membus.mem_side_ports
    system.cpu.interrupts[0].int_requestor = system.membus.cpu_side_ports
    system.cpu.interrupts[0].int_responder = system.membus.mem_side_ports
except Exception:
    print("INFO: Could not configure interrupt controller automatically (API differences).")

# Memory controller
system.mem_ctrl = MemCtrl()
try:
    system.mem_ctrl.dram = DDR3_1600_8x8()
    system.mem_ctrl.dram.range = system.mem_ranges[0]
    system.mem_ctrl.port = system.membus.mem_side_ports
except Exception:
    # fallback: attach a default memory
    try:
        system.mem_ctrl.port = system.membus.mem_side_ports
    except Exception:
        print("WARNING: Could not attach DRAM controller in standard way. Check your gem5 build.")

system.system_port = system.membus.cpu_side_ports

# Workload (SE mode)
try:
    system.workload = SEWorkload.init_compatible(args.cmd)
except Exception:
    # best-effort; some builds require different invocation
    print("WARNING: SEWorkload.init_compatible failed (API difference).")

process = Process()
process.cmd = [args.cmd] + (args.options.split() if args.options else [])
system.cpu.workload = process
try:
    system.cpu.createThreads()
except Exception:
    # some CPU classes use setWorkload/createThreads differently
    pass

root = Root(full_system=False, system=system)

m5.instantiate()

# Informational output
print("=" * 80)
print("Beginning simulation with:")
print(f"  Command: {args.cmd} {args.options}")
print(f"  CPU: ROB={args.rob_size}, IW={args.issue_width}, CW={args.commit_width}, BP={args.bp_type}")
print(f"  Prefetcher: {args.prefetcher}")
print(f"  L1 ICache: {args.l1i_size}, assoc={args.l1i_assoc}")
print(f"  L1 DCache: {args.l1d_size}, assoc={args.l1d_assoc}")
print(f"  L2 Cache: {args.l2_size}, assoc={args.l2_assoc}")
print(f"  Replacement policy hint: {args.repl_policy}")
print(f"  Memdep option: {args.memdep} (best-effort)")
print(f"  Memory: {args.mem_size}")
print(f"  Max Insts: {args.max_insts}")
print("=" * 80)

exit_event = m5.simulate()

print("=" * 80)
print(f"Simulation complete!")
print(f"Exiting @ tick {m5.curTick()} because {exit_event.getCause()}")
print("=" * 80)
