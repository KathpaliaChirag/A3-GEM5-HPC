#!/usr/bin/env python3
import argparse
import m5
from m5.objects import *

# -----------------------
# Minimal cache wrappers (provide required params)
# -----------------------
class L1Cache(Cache):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.assoc = 2
        self.tag_latency = 1
        self.data_latency = 1
        self.response_latency = 1
        self.mshrs = 4
        self.tgts_per_mshr = 20

class L2Cache(Cache):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.assoc = 8
        self.tag_latency = 8
        self.data_latency = 8
        self.response_latency = 8
        self.mshrs = 16
        self.tgts_per_mshr = 12

# -----------------------
# Argument parser
# -----------------------
parser = argparse.ArgumentParser(description="Custom gem5 run script for experiments")
parser.add_argument("--cmd", type=str, required=True, help="Path to benchmark binary (relative or absolute)")
parser.add_argument("--options", type=str, default="", help="Options to pass to the binary")
parser.add_argument("--cpu-type", type=str, default="DerivO3CPU", help="CPU type (DerivO3CPU supported)")
parser.add_argument("--rob-size", type=int, default=128, help="ROB size")
parser.add_argument("--issue-width", type=int, default=4, help="Issue width")
parser.add_argument("--commit-width", type=int, default=4, help="Commit width")
parser.add_argument("--l1d-size", type=str, default="32kB", help="L1 D-cache size")
parser.add_argument("--l1i-size", type=str, default="32kB", help="L1 I-cache size")
parser.add_argument("--l2-size", type=str, default="256kB", help="L2 cache size")
parser.add_argument("--bp-type", type=str, default="TournamentBP", help="Branch predictor type (TournamentBP,BiModeBP,LocalBP,LTAGE)")
parser.add_argument("--max-insts", type=int, default=1000000, help="Max instructions to simulate")
args = parser.parse_args()

# -----------------------
# System and clock
# -----------------------
system = System()
system.clk_domain = SrcClockDomain()
system.clk_domain.clock = "1GHz"
system.clk_domain.voltage_domain = VoltageDomain()

system.mem_mode = "timing"
system.mem_ranges = [AddrRange("512MB")]

# -----------------------
# CPU (DerivO3CPU)
# -----------------------
if args.cpu_type != "DerivO3CPU":
    print("Warning: script primarily supports DerivO3CPU; falling back may behave differently.")

system.cpu = DerivO3CPU()
# configure ROB and widths
system.cpu.numROBEntries = int(args.rob_size)
system.cpu.issueWidth = int(args.issue_width)
system.cpu.commitWidth = int(args.commit_width)

# branch predictor selection
bp = args.bp_type.upper()
if bp == "TOURNAMENTBP":
    system.cpu.branchPred = TournamentBP()
elif bp == "BIMODEBP":
    system.cpu.branchPred = BiModeBP()
elif bp == "LOCALBP":
    system.cpu.branchPred = LocalBP()
elif bp == "LTAGE":
    try:
        system.cpu.branchPred = LTAGE()
    except NameError:
        print("Warning: LTAGE predictor not available in this gem5 build; using TournamentBP instead.")
        system.cpu.branchPred = TournamentBP()
else:
    print(f"Unknown bp-type '{args.bp_type}', defaulting to TournamentBP")
    system.cpu.branchPred = TournamentBP()

# -----------------------
# Buses and caches
# -----------------------
# two-bus design: l2bus between L1 and L2; membus between L2 and memory
system.l2bus = SystemXBar()
system.membus = SystemXBar()

# L1s
system.cpu.icache = L1Cache(size=args.l1i_size)
system.cpu.dcache = L1Cache(size=args.l1d_size)

# L2
system.l2cache = L2Cache(size=args.l2_size)

# connect CPU <-> L1
# each L1 needs cpu_side and mem_side ports configured properly
system.cpu.icache.cpu_side = system.cpu.icache_port
system.cpu.dcache.cpu_side = system.cpu.dcache_port

# connect L1 -> L2 bus (mem_side of L1 -> cpu_side_ports of l2bus)
system.cpu.icache.mem_side = system.l2bus.cpu_side_ports
system.cpu.dcache.mem_side = system.l2bus.cpu_side_ports

# connect L2 to l2bus
system.l2cache.cpu_side = system.l2bus.mem_side_ports

# connect L2 to membus
system.l2cache.mem_side = system.membus.cpu_side_ports

# connect membus to memory controller later

# system port
system.system_port = system.membus.cpu_side_ports

# -----------------------
# Memory controller
# -----------------------
system.mem_ctrl = MemCtrl()
system.mem_ctrl.dram = DDR3_1600_8x8()
system.mem_ctrl.dram.range = system.mem_ranges[0]
system.mem_ctrl.port = system.membus.mem_side_ports

# -----------------------
# Interrupt controller (X86 O3 needs this)
# -----------------------
# create interrupt controller and robustly wire it to membus
try:
    system.cpu.createInterruptController()
except Exception:
    # some gem5 builds might raise; ignore if already present
    pass

# Connect interrupt ports robustly: different gem5 builds expose different attribute shapes
if hasattr(system.cpu, "interrupts"):
    intr = system.cpu.interrupts
    # intr can be a single object or an array/list in some builds
    if isinstance(intr, (list, tuple)):
        for idx, iobj in enumerate(intr):
            try:
                iobj.pio = system.membus.mem_side_ports
            except Exception:
                pass
            try:
                # int_requestor / int_responder names vary; do best-effort
                if hasattr(iobj, "int_requestor"):
                    iobj.int_requestor = system.membus.cpu_side_ports
            except Exception:
                pass
            try:
                if hasattr(iobj, "int_responder"):
                    iobj.int_responder = system.membus.mem_side_ports
            except Exception:
                pass
    else:
        try:
            intr.pio = system.membus.mem_side_ports
        except Exception:
            pass
        try:
            if hasattr(intr, "int_requestor"):
                intr.int_requestor = system.membus.cpu_side_ports
        except Exception:
            pass
        try:
            if hasattr(intr, "int_responder"):
                intr.int_responder = system.membus.mem_side_ports
        except Exception:
            pass

# As an extra safety, attempt to set named ports if those exist
# Some versions require explicit int_requestor/int_responder attributes on cpu itself
try:
    if hasattr(system.cpu, "int_requestor"):
        system.cpu.int_requestor = system.membus.cpu_side_ports
except Exception:
    pass
try:
    if hasattr(system.cpu, "int_responder"):
        system.cpu.int_responder = system.membus.mem_side_ports
except Exception:
    pass

# -----------------------
# Workload (SE mode)
# -----------------------
# Inform gem5 of the workload and attach process to CPU
system.workload = SEWorkload.init_compatible(args.cmd)

proc = Process()
proc.executable = args.cmd
proc.cmd = [args.cmd] + (args.options.split() if args.options else [])
system.cpu.workload = proc
system.cpu.createThreads()

# -----------------------
# Instantiate and run
# -----------------------
root = Root(full_system=False, system=system)
m5.instantiate()

print("Beginning simulation with:")
print(f"  cmd = {args.cmd} options='{args.options}'")
print(f"  ROB={args.rob_size} issue={args.issue_width} commit={args.commit_width} L1d={args.l1d_size} L1i={args.l1i_size} L2={args.l2_size} BP={args.bp_type}")

exit_event = m5.simulate(args.max_insts)
print("Exiting @ tick {} because {}".format(m5.curTick(), exit_event.getCause()))
