import sys

def parse_stats(stats_file):
    stats = {}
    with open(stats_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            name, value = parts[0], parts[1]
            # Try to parse value
            try:
                stats[name] = float(value)
            except ValueError:
                continue
    
    # Collect desired stats
    print(f"\nResults from: {stats_file}")
    print(f"  Instructions executed: {int(stats.get('simInsts', 0))}")
    print(f"  IPC: {stats.get('system.cpu.ipc', None)}")
    print(f"  CPI: {stats.get('system.cpu.cpi', None)}")

    # L1D miss rate
    l1d_miss = stats.get("system.cpu.dcache.overallMissRate::total", None)
    print(f"  L1D miss rate: {l1d_miss}")

    # L2 miss rate (if present)
    l2_miss = stats.get("system.l2.overallMissRate::total", None)
    print(f"  L2 miss rate: {l2_miss}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 parse_stats.py <stats_file> [<stats_file2> ...]")
        sys.exit(1)

    for stats_file in sys.argv[1:]:
        parse_stats(stats_file)
