"""
debug_drop_rate.py - Scientific investigation of high drop rates

This script runs controlled experiments to identify why generated workloads
have 66% drop rate while the original has 0%.

Experiments:
1. Compare workload characteristics (bytes/sec, packet sizes, etc.)
2. Test original workload with random critical flows
3. Test generated workload with original's critical flows
4. Analyze traffic distribution across network links
"""

import pandas as pd
import numpy as np
import subprocess
import json
from pathlib import Path
import sys


def load_workload(filepath: str) -> pd.DataFrame:
    """Load and parse a workload CSV."""
    return pd.read_csv(filepath)


def analyze_workload(df: pd.DataFrame, name: str) -> dict:
    """Compute workload statistics."""
    duration = df['start'].max() - df['start'].min()
    total_bytes = df['psize'].sum()
    
    stats = {
        'name': name,
        'num_packets': len(df),
        'num_flows': df['link'].nunique(),
        'duration_sec': duration,
        'total_bytes_gb': total_bytes / 1e9,
        'bytes_per_sec_gbps': (total_bytes * 8) / duration / 1e9,  # Gbps
        'avg_packet_mb': df['psize'].mean() / 1e6,
        'max_packet_mb': df['psize'].max() / 1e6,
        'min_packet_mb': df['psize'].min() / 1e6,
        'std_packet_mb': df['psize'].std() / 1e6,
        'packets_per_sec': len(df) / duration,
    }
    
    # Timing analysis
    df_sorted = df.sort_values('start')
    inter_arrival = df_sorted['start'].diff().dropna()
    stats['avg_inter_arrival_sec'] = inter_arrival.mean()
    stats['min_inter_arrival_sec'] = inter_arrival.min()
    
    # Check for bursts (multiple packets within 0.5s)
    burst_threshold = 0.5
    bursts = (inter_arrival < burst_threshold).sum()
    stats['num_bursts'] = bursts
    
    return stats


def run_cloudsim(workload_file: str, critical_flows_file: str, output_dir: str) -> dict:
    """Run CloudSim and return episode summary."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    cmd = f'mvn exec:java -q "-Dexec.mainClass=org.cloudbus.cloudsim.sdn.example.CFRRLTrainingRunner" "-Dexec.args={workload_file} {critical_flows_file} {output_dir}/"'
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
    
    summary_file = Path(output_dir) / 'episode_summary.json'
    if summary_file.exists():
        with open(summary_file) as f:
            return json.load(f)
    return None


def create_critical_flows_file(flow_ids: list, filepath: str):
    """Create a critical flows file."""
    with open(filepath, 'w') as f:
        for fid in flow_ids:
            f.write(f"{fid}\n")


def extract_flow_ids(df: pd.DataFrame) -> list:
    """Extract flow IDs from workload."""
    return [int(f.replace('flow_', '')) for f in df['link'].unique()]


def main():
    print("=" * 70)
    print("DROP RATE INVESTIGATION")
    print("=" * 70)
    
    # Load workloads
    original = load_workload('dataset-abilene/abilene-workload.csv')
    
    # Check if generated workload exists from training
    gen_paths = list(Path('.').glob('test_duration_fixed/episodes/*/workload.csv'))
    if not gen_paths:
        gen_paths = list(Path('.').glob('training_outputs/episodes/*/workload.csv'))
    
    if gen_paths:
        generated = load_workload(str(gen_paths[0]))
    else:
        print("No generated workload found. Generate one first with training.")
        return
    
    # =========================================================================
    # EXPERIMENT 1: Compare workload characteristics
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Workload Characteristics Comparison")
    print("=" * 70)
    
    orig_stats = analyze_workload(original, "Original")
    gen_stats = analyze_workload(generated, "Generated")
    
    print(f"\n{'Metric':<30} {'Original':>15} {'Generated':>15} {'Diff':>15}")
    print("-" * 75)
    
    for key in orig_stats:
        if key == 'name':
            continue
        orig_val = orig_stats[key]
        gen_val = gen_stats[key]
        if isinstance(orig_val, float):
            diff = gen_val - orig_val
            print(f"{key:<30} {orig_val:>15.3f} {gen_val:>15.3f} {diff:>+15.3f}")
        else:
            print(f"{key:<30} {orig_val:>15} {gen_val:>15}")
    
    # =========================================================================
    # EXPERIMENT 2: Original workload with RANDOM critical flows
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Original workload with RANDOM critical flows")
    print("=" * 70)
    print("(Does drop rate depend on WHICH flows are critical?)")
    
    orig_flows = extract_flow_ids(original)
    
    # Create random critical flows
    np.random.seed(42)
    random_critical = np.random.choice(orig_flows, size=8, replace=False).tolist()
    create_critical_flows_file(random_critical, 'debug_random_critical.txt')
    print(f"Random critical flows: {random_critical}")
    
    result = run_cloudsim(
        'dataset-abilene/abilene-workload.csv',
        'debug_random_critical.txt',
        'debug_outputs/exp2_orig_random_critical'
    )
    
    if result:
        print(f"Result: drop_rate = {result['drop_rate']*100:.1f}%, mean_queue = {result['mean_queuing_ms']:.1f}ms")
    else:
        print("CloudSim failed to run")
    
    # =========================================================================
    # EXPERIMENT 3: Generated workload with ALL flows critical
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Generated workload with ALL flows critical (K=50)")
    print("=" * 70)
    print("(If all flows get LP-optimized routing, do drops decrease?)")
    
    gen_flows = extract_flow_ids(generated)
    create_critical_flows_file(gen_flows, 'debug_all_critical.txt')
    print(f"All {len(gen_flows)} flows marked as critical")
    
    result = run_cloudsim(
        str(gen_paths[0]),
        'debug_all_critical.txt',
        'debug_outputs/exp3_gen_all_critical'
    )
    
    if result:
        print(f"Result: drop_rate = {result['drop_rate']*100:.1f}%, mean_queue = {result['mean_queuing_ms']:.1f}ms")
    else:
        print("CloudSim failed to run")
    
    # =========================================================================
    # EXPERIMENT 4: Test with FEWER packets
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Reduce traffic load")
    print("=" * 70)
    print("(Is the network simply overloaded?)")
    
    # Create a lighter workload from generated (first 25 packets only)
    light = generated.head(25).copy()
    light.to_csv('debug_light_workload.csv', index=False)
    
    light_flows = extract_flow_ids(light)
    create_critical_flows_file(light_flows[:8], 'debug_light_critical.txt')
    
    result = run_cloudsim(
        'debug_light_workload.csv',
        'debug_light_critical.txt',
        'debug_outputs/exp4_light_workload'
    )
    
    if result:
        print(f"Result with 25 packets: drop_rate = {result['drop_rate']*100:.1f}%, mean_queue = {result['mean_queuing_ms']:.1f}ms")
    else:
        print("CloudSim failed to run")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY & NEXT STEPS")
    print("=" * 70)
    
    print("""
Based on the experiments above, determine:

1. If Exp 2 shows HIGH drops → Critical flow selection matters
   → Need to select better critical flows

2. If Exp 3 shows LOW drops → LP routing helps
   → Problem is that random K=8 isn't enough coverage

3. If Exp 3 shows HIGH drops → LP routing doesn't help
   → Problem is traffic intensity or distribution

4. If Exp 4 shows LOW drops → Network is overloaded
   → Need to reduce traffic or increase capacity
    """)
    
    # Cleanup
    for f in ['debug_random_critical.txt', 'debug_all_critical.txt', 
              'debug_light_workload.csv', 'debug_light_critical.txt']:
        Path(f).unlink(missing_ok=True)


if __name__ == "__main__":
    main()