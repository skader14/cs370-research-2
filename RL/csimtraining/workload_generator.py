"""
workload_generator.py - Generate random workloads for CloudSim training episodes.

Each episode gets a unique random workload to prevent overfitting and ensure
the policy generalizes across different traffic patterns.

CloudSim workload format (CSV with header):
    start,source,z,w1,link,dest,psize,w2
    0.4,vm_7,0,1,flow_84,vm_8,146676818,1
    
Where:
    - start: When packet transmission starts (seconds)
    - source: Source VM name (e.g., "vm_7")
    - z: Always 0
    - w1: Always 1
    - link: Flow name (e.g., "flow_84")
    - dest: Destination VM name (e.g., "vm_8")
    - psize: Packet size in bytes
    - w2: Always 1

IMPORTANT FINDINGS:
1. Original Abilene workload has 1 packet per flow. Multiple packets
   per flow causes high drop rates due to buffer overflow.
2. Only certain flow IDs work well with the Abilene topology.
   Random flow ID selection causes 66% packet drops!
   The original workload's 51 flow IDs achieve ~0% drops.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List
import os


# Abilene topology constants
NUM_NODES = 12
NUM_FLOWS = NUM_NODES * (NUM_NODES - 1)  # 132 flows

# CRITICAL: These are the "known good" flow IDs from the original Abilene workload.
# These flow IDs distribute traffic well across the topology and achieve ~0% drops.
# Random selection from all 132 flows causes ~66% drops due to bottleneck congestion.
ABILENE_GOOD_FLOWS = [
    5, 7, 9, 10, 12, 13, 14, 15, 17, 18, 22, 23, 27, 30, 31, 33, 37, 39, 41, 
    47, 48, 49, 50, 52, 62, 72, 74, 77, 81, 83, 84, 85, 88, 89, 95, 96, 98, 
    102, 103, 106, 108, 109, 110, 113, 114, 115, 116, 122, 123, 130, 131
]

# "Stress" flows - these are NOT in the good flows list and may cause congestion
# Used for mixed difficulty training so the policy learns to prioritize
ABILENE_STRESS_FLOWS = [f for f in range(NUM_FLOWS) if f not in ABILENE_GOOD_FLOWS]


def get_flow_pool(difficulty: str = 'easy', num_stress: int = 15, seed: int = None) -> List[int]:
    """
    Get a flow pool based on difficulty level.
    
    Args:
        difficulty: 'easy' (only good flows), 'mixed' (good + some stress), 'hard' (all flows)
        num_stress: Number of stress flows to include for 'mixed' difficulty
        seed: Random seed for selecting stress flows
    
    Returns:
        List of flow IDs to sample from
    """
    if difficulty == 'easy':
        return ABILENE_GOOD_FLOWS.copy()
    elif difficulty == 'mixed':
        rng = np.random.default_rng(seed)
        stress_sample = rng.choice(ABILENE_STRESS_FLOWS, size=min(num_stress, len(ABILENE_STRESS_FLOWS)), replace=False).tolist()
        return ABILENE_GOOD_FLOWS + stress_sample
    elif difficulty == 'hard':
        return list(range(NUM_FLOWS))
    else:
        raise ValueError(f"Unknown difficulty: {difficulty}. Use 'easy', 'mixed', or 'hard'.")


def flow_id_to_nodes(flow_id: int) -> Tuple[int, int]:
    """Convert flow ID to (src, dst) node pair."""
    src = flow_id // (NUM_NODES - 1)
    dst_idx = flow_id % (NUM_NODES - 1)
    dst = dst_idx if dst_idx < src else dst_idx + 1
    return src, dst


def nodes_to_flow_id(src: int, dst: int) -> int:
    """Convert (src, dst) node pair to flow ID."""
    dst_idx = dst if dst < src else dst - 1
    return src * (NUM_NODES - 1) + dst_idx


def generate_workload(
    num_packets: int = 50,                  # Target packet count (~51 in original)
    duration: float = 20.0,                 # Match original (~20s)
    seed: Optional[int] = None,
    min_packet_size: int = 20_000_000,      # 20 MB (original min: 13 MB)
    max_packet_size: int = 400_000_000,     # 400 MB (original max: 957 MB)
    use_good_flows: bool = True,            # DEPRECATED: use difficulty instead
    difficulty: str = 'easy',               # 'easy', 'mixed', or 'hard'
    flow_pool: Optional[List[int]] = None,  # Custom flow pool (overrides difficulty)
) -> pd.DataFrame:
    """
    Generate a random workload for one training episode.
    
    Difficulty levels:
    - 'easy': Only good flows (51 flows that work well with topology)
    - 'mixed': Good flows + 15 stress flows (policy must learn to prioritize)
    - 'hard': All 132 flows (high drop rate, for stress testing)
    
    Args:
        num_packets: Number of packets (= number of active flows)
        duration: Simulation duration in seconds
        seed: Random seed (None for truly random)
        min_packet_size: Minimum packet size in bytes
        max_packet_size: Maximum packet size in bytes
        use_good_flows: DEPRECATED, use difficulty instead
        difficulty: 'easy', 'mixed', or 'hard'
        flow_pool: Custom list of flow IDs to sample from
    
    Returns:
        DataFrame in CloudSim format with columns:
        start, source, z, w1, link, dest, psize, w2
    """
    rng = np.random.default_rng(seed)
    
    packets = []
    
    # Determine flow pool
    if flow_pool is not None:
        available_flows = flow_pool
    elif not use_good_flows:
        # Backward compatibility: use_good_flows=False means 'hard'
        available_flows = list(range(NUM_FLOWS))
    else:
        # Use difficulty setting
        available_flows = get_flow_pool(difficulty, seed=seed)
    
    # Select which flows are active (each gets exactly 1 packet)
    num_active = min(num_packets, len(available_flows))
    active_flows = rng.choice(available_flows, size=num_active, replace=False).tolist()
    
    for flow_id in active_flows:
        src, dst = flow_id_to_nodes(flow_id)
        
        # Random start time spread across duration
        start_time = rng.uniform(0, duration * 0.95)
        
        # Packet size (log-uniform for realistic distribution)
        log_min = np.log(min_packet_size)
        log_max = np.log(max_packet_size)
        packet_size = int(np.exp(rng.uniform(log_min, log_max)))
        
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
    
    # Create DataFrame and sort by start time
    df = pd.DataFrame(packets)
    df = df.sort_values('start').reset_index(drop=True)
    
    return df


def save_workload(df: pd.DataFrame, filepath: str) -> None:
    """
    Save workload to CSV in CloudSim format.
    
    CloudSim expects header row:
    start,source,z,w1,link,dest,psize,w2
    """
    # Ensure directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Save with header (CloudSim format)
    df.to_csv(filepath, index=False)


def load_workload(filepath: str) -> pd.DataFrame:
    """Load workload from CSV."""
    return pd.read_csv(filepath)


def get_workload_stats(df: pd.DataFrame) -> dict:
    """Get summary statistics for a workload."""
    # Extract flow_id from 'link' column (e.g., "flow_84" -> 84)
    flow_ids = df['link'].str.replace('flow_', '').astype(int)
    
    return {
        'num_packets': len(df),
        'num_active_flows': flow_ids.nunique(),
        'total_bytes': df['psize'].sum(),
        'mean_packet_size': df['psize'].mean(),
        'duration': df['start'].max() - df['start'].min(),
        'packets_per_flow': len(df) / max(1, flow_ids.nunique()),
    }


def generate_demand_vector(df: pd.DataFrame) -> np.ndarray:
    """
    Convert workload to demand vector for feature computation.
    
    Returns:
        Array of shape (132,) with total bytes per flow
    """
    demand = np.zeros(NUM_FLOWS)
    
    # Extract flow_id from 'link' column
    for _, row in df.iterrows():
        flow_id = int(row['link'].replace('flow_', ''))
        demand[flow_id] += row['psize']
    
    return demand


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    # Test workload generation
    print("Testing workload generation...")
    print("=" * 60)
    print(f"Using ABILENE_GOOD_FLOWS: {len(ABILENE_GOOD_FLOWS)} known-good flow IDs")
    
    # Generate a few workloads
    for i in range(3):
        df = generate_workload(num_packets=50, duration=20, seed=None)
        stats = get_workload_stats(df)
        print(f"\nWorkload {i+1}:")
        print(f"  Packets: {stats['num_packets']}")
        print(f"  Active flows: {stats['num_active_flows']}")
        print(f"  Packets per flow: {stats['packets_per_flow']:.1f}")
        print(f"  Total bytes: {stats['total_bytes'] / 1e9:.2f} GB")
        print(f"  Avg packet size: {stats['mean_packet_size'] / 1e6:.1f} MB")
        print(f"  Duration: {stats['duration']:.1f}s")
    
    # Test saving
    df = generate_workload(num_packets=50, duration=20, seed=42)
    save_workload(df, "test_workload.csv")
    print("\n" + "=" * 60)
    print("Saved test_workload.csv")
    print("\nFirst 5 rows:")
    print(df.head().to_string())
    
    # Show which flow IDs were selected
    flow_ids = sorted([int(f.replace('flow_', '')) for f in df['link'].unique()])
    print(f"\nFlow IDs used (from ABILENE_GOOD_FLOWS):")
    print(f"  {flow_ids[:10]}... ({len(flow_ids)} total)")
    
    # Compare to original
    print("\n" + "=" * 60)
    print("COMPARISON TO ORIGINAL ABILENE:")
    print("-" * 40)
    print(f"{'Metric':<25} {'Original':<15} {'Generated':<15}")
    print("-" * 40)
    print(f"{'Packets':<25} {'51':<15} {len(df):<15}")
    print(f"{'Packets per flow':<25} {'1.0':<15} {stats['packets_per_flow']:<15.1f}")
    print(f"{'Total GB':<25} {'6.39':<15} {stats['total_bytes']/1e9:<15.2f}")
    print(f"{'Duration (s)':<25} {'20':<15} {stats['duration']:<15.1f}")
    
    # Test demand vector
    demand = generate_demand_vector(df)
    print(f"\nDemand vector: {np.count_nonzero(demand)} non-zero flows")
    print(f"Max demand: {demand.max() / 1e9:.2f} GB")
    print(f"Total demand: {demand.sum() / 1e9:.2f} GB")