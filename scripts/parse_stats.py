import sys

def parse_stats(stats_file):
    stats = {}
    with open(stats_file) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            name, value = parts[0], parts[1]
            try:
                stats[name] = float(value)
            except ValueError:
                continue

    # Collect key stats
    instructions = stats.get("simInsts")
    ipc = stats.get("system.cpu.ipc")
    cpi = stats.get("system.cpu.cpi")

    # L1D miss rate
    l1d_miss_rate = stats.get("system.cpu.dcache.overallMissRate::total")

    # L2 miss rate (computed manually)
    l2_misses = stats.get("system.l2cache.overallMisses::total")
    l2_accesses = stats.get("system.l2cache.overallAccesses::total")
    l2_miss_rate = None
    if l2_misses is not None and l2_accesses and l2_accesses > 0:
        l2_miss_rate = l2_misses / l2_accesses

    print(f"\nResults from: {stats_file}")
    print(f"  Instructions executed: {instructions}")
    print(f"  IPC: {ipc}")
    print(f"  CPI: {cpi}")
    print(f"  L1D miss rate: {l1d_miss_rate}")
    print(f"  L2 miss rate: {l2_miss_rate}")

if __name__ == "__main__":
    for stats_file in sys.argv[1:]:
        parse_stats(stats_file)
