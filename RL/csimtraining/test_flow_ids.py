"""
test_flow_ids.py - Definitive test: are the FLOW IDs the problem?

This script tests whether using the ORIGINAL workload's flow IDs
(but with new random timing and packet sizes) produces 0% drops.

If YES (0% drops): Problem is which flow IDs we select
If NO (high drops): Problem is something else (timing, sizes, etc.)
"""

import pandas as pd
import numpy as np
import subprocess
import json
from pathlib import Path


def flow_id_to_nodes(flow_id: int) -> tuple:
    """Convert flow ID to (src, dst) node pair."""
    NUM_NODES = 12
    src = flow_id // (NUM_NODES - 1)
    dst_idx = flow_id % (NUM_NODES - 1)
    dst = dst_idx if dst_idx < src else dst_idx + 1
    return src, dst


def generate_workload_with_flow_ids(flow_ids: list, duration: float = 20.0, seed: int = None) -> pd.DataFrame:
    """Generate workload using specific flow IDs with random timing/sizes."""
    rng = np.random.default_rng(seed)
    packets = []
    
    for flow_id in flow_ids:
        src, dst = flow_id_to_nodes(flow_id)
        start_time = rng.uniform(0, duration * 0.95)
        # Use similar size distribution to original (log-uniform 20MB-400MB)
        packet_size = int(np.exp(rng.uniform(np.log(20e6), np.log(400e6))))
        
        packets.append({
            'start': round(start_time, 1),
            'source': f'vm_{src}',
            'z': 0,
            'w1': 1,
            'link': f'flow_{flow_id}',
            'dest': f'vm_{dst}',
            'psize': packet_size,
            'w2': 1
        })
    
    df = pd.DataFrame(packets).sort_values('start').reset_index(drop=True)
    return df


def run_cloudsim(workload_file: str, critical_file: str, output_dir: str) -> dict:
    """Run CloudSim and return results."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    cmd = f'mvn exec:java -q "-Dexec.mainClass=org.cloudbus.cloudsim.sdn.example.CFRRLTrainingRunner" "-Dexec.args={workload_file} {critical_file} {output_dir}/"'
    
    result = subprocess.run(cmd, shell=True, capture_output=True, timeout=120)
    
    summary_file = Path(output_dir) / 'episode_summary.json'
    if summary_file.exists():
        with open(summary_file) as f:
            return json.load(f)
    return None


def main():
    print("=" * 70)
    print("DEFINITIVE TEST: Are the Flow IDs the Problem?")
    print("=" * 70)
    
    # Load original workload to get its flow IDs
    original = pd.read_csv('dataset-abilene/abilene-workload.csv')
    original_flow_ids = [int(f.replace('flow_', '')) for f in original['link'].unique()]
    
    print(f"\nOriginal workload flow IDs ({len(original_flow_ids)} flows):")
    print(f"  {sorted(original_flow_ids)}")
    
    # Load a generated workload to get its flow IDs
    gen_paths = list(Path('.').glob('test_duration_fixed/episodes/*/workload.csv'))
    if not gen_paths:
        gen_paths = list(Path('.').glob('training_*/episodes/*/workload.csv'))
    
    if gen_paths:
        generated = pd.read_csv(str(gen_paths[0]))
        generated_flow_ids = [int(f.replace('flow_', '')) for f in generated['link'].unique()]
        print(f"\nGenerated workload flow IDs ({len(generated_flow_ids)} flows):")
        print(f"  {sorted(generated_flow_ids)}")
        
        # Show overlap
        overlap = set(original_flow_ids) & set(generated_flow_ids)
        print(f"\nOverlap: {len(overlap)} flows in common")
        print(f"  {sorted(overlap)}")
    
    # =========================================================================
    # TEST 1: Original flow IDs with NEW random timing/sizes
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 1: Original flow IDs + Random timing/sizes")
    print("=" * 70)
    
    test_workload = generate_workload_with_flow_ids(original_flow_ids, seed=42)
    test_workload.to_csv('test_orig_ids.csv', index=False)
    
    # Create critical flows file (first 8)
    with open('test_orig_ids_critical.txt', 'w') as f:
        for fid in original_flow_ids[:8]:
            f.write(f"{fid}\n")
    
    print(f"Generated new workload with original's {len(original_flow_ids)} flow IDs")
    print(f"Packet sizes: {test_workload['psize'].min()/1e6:.1f} - {test_workload['psize'].max()/1e6:.1f} MB")
    print(f"Timing: {test_workload['start'].min():.1f} - {test_workload['start'].max():.1f} seconds")
    
    result1 = run_cloudsim('test_orig_ids.csv', 'test_orig_ids_critical.txt', 'debug_outputs/test1_orig_ids')
    
    if result1:
        print(f"\n>>> Result: drop_rate = {result1['drop_rate']*100:.1f}%, mean_queue = {result1['mean_queuing_ms']:.1f}ms")
    
    # =========================================================================
    # TEST 2: Generated flow IDs with ORIGINAL's timing pattern
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 2: Generated flow IDs + Original's timing distribution")
    print("=" * 70)
    
    if gen_paths:
        # Copy timing from original, but use generated flow IDs
        test2_packets = []
        for i, (_, row) in enumerate(original.iterrows()):
            if i < len(generated_flow_ids):
                flow_id = generated_flow_ids[i]
                src, dst = flow_id_to_nodes(flow_id)
                test2_packets.append({
                    'start': row['start'],  # Original timing
                    'source': f'vm_{src}',
                    'z': 0,
                    'w1': 1,
                    'link': f'flow_{flow_id}',
                    'dest': f'vm_{dst}',
                    'psize': row['psize'],  # Original size
                    'w2': 1
                })
        
        test2_df = pd.DataFrame(test2_packets)
        test2_df.to_csv('test_gen_ids_orig_timing.csv', index=False)
        
        with open('test_gen_ids_critical.txt', 'w') as f:
            for fid in generated_flow_ids[:8]:
                f.write(f"{fid}\n")
        
        result2 = run_cloudsim('test_gen_ids_orig_timing.csv', 'test_gen_ids_critical.txt', 'debug_outputs/test2_gen_ids')
        
        if result2:
            print(f"\n>>> Result: drop_rate = {result2['drop_rate']*100:.1f}%, mean_queue = {result2['mean_queuing_ms']:.1f}ms")
    
    # =========================================================================
    # ANALYSIS
    # =========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    if result1 and result1['drop_rate'] < 0.1:
        print("\n✓ TEST 1 PASSED (low drops with original flow IDs)")
        print("  → The FLOW IDs matter! Original's flow distribution is special.")
        
        if result2 and result2['drop_rate'] > 0.3:
            print("\n✗ TEST 2 FAILED (high drops with generated flow IDs)")
            print("  → CONFIRMED: Problem is which (src,dst) pairs we're selecting")
            print("\n  SOLUTION: Use original's flow ID distribution as template")
        
    elif result1 and result1['drop_rate'] > 0.3:
        print("\n✗ TEST 1 FAILED (high drops even with original flow IDs)")
        print("  → Problem is NOT just flow IDs")
        print("  → Check original workload's exact packet sizes and timing")
    
    # Cleanup
    for f in ['test_orig_ids.csv', 'test_orig_ids_critical.txt', 
              'test_gen_ids_orig_timing.csv', 'test_gen_ids_critical.txt']:
        Path(f).unlink(missing_ok=True)


if __name__ == "__main__":
    main()