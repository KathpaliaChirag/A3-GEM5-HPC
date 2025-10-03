import m5
from m5.objects import *
import argparse

# ----------------------------
# Parse command-line arguments
# ----------------------------
parser = argparse.ArgumentParser(description="Custom gem5 simulation configuration")

parser.add_argument("--cmd", type=str, required=True,
                    help="Command to run")
parser.add_argument("--options", type=str, default="",
                    help="Options for the benchmark")
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
                    choices=["LocalBP", "BiModeBP", "TournamentBP", "LTAGE"],
                    help="Branch predictor type (default: TournamentBP)")
parser.add_argument("--prefetcher", type=str, default="none",
                    choices=["none", "stride", "tagged"],
                    help="Prefetcher type (default: none)")
parser.add_argument("--memdep", type=str, default="none",
                    choices=["none", "simple", "storeset"],
                    help="Memory dependence predictor type (default: none)")
parser.add_argument("--max-insts", type=int, default=1000000,
                    help="Maximum number of instructions to simulate (default: 1000000)")
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

# CPU
system.cpu = DerivO3CPU()
system.cpu.numROBEntries = args.rob_size
system.cpu.fetchWidth = args.issue_width
system.cpu.decodeWidth = args.issue_width
system.cpu.renameWidth = args.issue_width
system.cpu.issueWidth = args.issue_width
system.cpu.wbWidth = args.commit_width
system.cpu.commitWidth = args.commit_width
system.cpu.squashWidth = args.commit_width

# Set max instructions
system.cpu.max_insts_any_thread = args.max_insts

# Branch predictor
if args.bp_type == "LocalBP":
    system.cpu.branchPred = LocalBP()
elif args.bp_type == "BiModeBP":
    system.cpu.branchPred = BiModeBP()
elif args.bp_type == "TournamentBP":
    system.cpu.branchPred = TournamentBP()
elif args.bp_type == "LTAGE":
    system.cpu.branchPred = LTAGE()

# ----------------------------
# Helper functions for extras
# ----------------------------
def make_prefetcher(pf_type):
    if pf_type == "none":
        return None
    elif pf_type == "stride":
        return StridePrefetcher()
    elif pf_type == "tagged":
        return TaggedPrefetcher()
    else:
        raise ValueError(f"Unknown prefetcher: {pf_type}")

def make_memdep(md_type):
    if md_type == "none":
        return None
    elif md_type == "simple":
        return SimpleMemDepPred()
    elif md_type == "storeset":
        return StoreSet()
    else:
        raise ValueError(f"Unknown memdep predictor: {md_type}")

# ----------------------------
# Caches
# ----------------------------
system.cpu.icache = Cache(size=args.l1i_size, assoc=args.l1i_assoc,
                          tag_latency=2, data_latency=2, response_latency=2,
                          mshrs=4, tgts_per_mshr=20)

system.cpu.dcache = Cache(size=args.l1d_size, assoc=args.l1d_assoc,
                          tag_latency=2, data_latency=2, response_latency=2,
                          mshrs=4, tgts_per_mshr=20)

system.l2cache = Cache(size=args.l2_size, assoc=args.l2_assoc,
                       tag_latency=20, data_latency=20, response_latency=20,
                       mshrs=20, tgts_per_mshr=12)

# Prefetcher
pf = make_prefetcher(args.prefetcher)
if pf:
    system.cpu.icache.prefetcher = pf
    system.cpu.dcache.prefetcher = pf
    system.l2cache.prefetcher = pf

# MemDep predictor
md = make_memdep(args.memdep)
if md:
    system.cpu.memDepPred = md

# ----------------------------
# Buses
# ----------------------------
system.l2bus = L2XBar()
system.membus = SystemXBar()

system.cpu.icache.cpu_side = system.cpu.icache_port
system.cpu.dcache.cpu_side = system.cpu.dcache_port
system.cpu.icache.mem_side = system.l2bus.cpu_side_ports
system.cpu.dcache.mem_side = system.l2bus.cpu_side_ports
system.l2cache.cpu_side = system.l2bus.mem_side_ports
system.l2cache.mem_side = system.membus.cpu_side_ports

# Interrupts
system.cpu.createInterruptController()
system.cpu.interrupts[0].pio = system.membus.mem_side_ports
system.cpu.interrupts[0].int_requestor = system.membus.cpu_side_ports
system.cpu.interrupts[0].int_responder = system.membus.mem_side_ports

# Memory
system.mem_ctrl = MemCtrl()
system.mem_ctrl.dram = DDR3_1600_8x8()
system.mem_ctrl.dram.range = system.mem_ranges[0]
system.mem_ctrl.port = system.membus.mem_side_ports

system.system_port = system.membus.cpu_side_ports

# Workload
system.workload = SEWorkload.init_compatible(args.cmd)
process = Process()
process.cmd = [args.cmd] + (args.options.split() if args.options else [])
system.cpu.workload = process
system.cpu.createThreads()

# Root
root = Root(full_system=False, system=system)
m5.instantiate()

print("=" * 80)
print("Beginning simulation with:")
print(f"  Command: {args.cmd} {args.options}")
print(f"  CPU Configuration:")
print(f"    - Clock: {args.cpu_clock}")
print(f"    - ROB Size: {args.rob_size}")
print(f"    - Issue Width: {args.issue_width}")
print(f"    - Commit Width: {args.commit_width}")
print(f"    - Branch Predictor: {args.bp_type}")
print(f"    - MemDep Predictor: {args.memdep}")
print(f"  Cache Configuration:")
print(f"    - L1 ICache: {args.l1i_size} (assoc: {args.l1i_assoc}) pf: {args.prefetcher}")
print(f"    - L1 DCache: {args.l1d_size} (assoc: {args.l1d_assoc}) pf: {args.prefetcher}")
print(f"    - L2 Cache: {args.l2_size} (assoc: {args.l2_assoc}) pf: {args.prefetcher}")
print(f"  Memory: {args.mem_size}")
print(f"  Max Instructions: {args.max_insts}")
print("=" * 80)

# Run
exit_event = m5.simulate()

print("=" * 80)
print(f"Simulation complete!")
print(f"Exiting @ tick {m5.curTick()} because {exit_event.getCause()}")
print("=" * 80)
