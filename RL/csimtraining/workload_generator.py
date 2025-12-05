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
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
import os


# Abilene topology constants
NUM_NODES = 12
NUM_FLOWS = NUM_NODES * (NUM_NODES - 1)  # 132 flows


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
    num_packets: int = 300,
    duration: float = 90.0,
    seed: Optional[int] = None,
    min_packet_size: int = 1_000_000,      # 1 MB
    max_packet_size: int = 500_000_000,    # 500 MB
    flow_activity_prob: float = 0.4,        # Probability a flow is active
    burst_prob: float = 0.1,                # Probability of burst traffic
    burst_multiplier: float = 3.0,          # Burst size multiplier
) -> pd.DataFrame:
    """
    Generate a random workload for one training episode.
    
    Args:
        num_packets: Target number of packets (approximate)
        duration: Simulation duration in seconds
        seed: Random seed (None for truly random)
        min_packet_size: Minimum packet size in bytes
        max_packet_size: Maximum packet size in bytes
        flow_activity_prob: Probability each flow has traffic
        burst_prob: Probability of a burst (multiple packets same flow)
        burst_multiplier: How many extra packets in a burst
    
    Returns:
        DataFrame in CloudSim format with columns:
        start, source, z, w1, link, dest, psize, w2
    """
    rng = np.random.default_rng(seed)
    
    packets = []
    
    # Determine which flows are active this episode
    active_flows = []
    for flow_id in range(NUM_FLOWS):
        if rng.random() < flow_activity_prob:
            active_flows.append(flow_id)
    
    # Ensure at least some flows are active
    if len(active_flows) < 10:
        active_flows = rng.choice(NUM_FLOWS, size=20, replace=False).tolist()
    
    # Generate packets
    packets_per_flow = max(1, num_packets // len(active_flows))
    
    for flow_id in active_flows:
        # Convert flow_id to src/dst nodes
        src, dst = flow_id_to_nodes(flow_id)
        
        # Number of packets for this flow (with some variance)
        n_packets = max(1, int(rng.poisson(packets_per_flow)))
        
        # Check for burst
        if rng.random() < burst_prob:
            n_packets = int(n_packets * burst_multiplier)
        
        for _ in range(n_packets):
            # Random start time within duration
            start_time = rng.uniform(0, duration * 0.9)  # Leave 10% buffer at end
            
            # Packet size (log-uniform for more realistic distribution)
            log_min = np.log(min_packet_size)
            log_max = np.log(max_packet_size)
            packet_size = int(np.exp(rng.uniform(log_min, log_max)))
            
            packets.append({
                'start': round(start_time, 1),  # Round to 0.1s precision
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
    
    # Generate a few workloads
    for i in range(3):
        df = generate_workload(num_packets=300, duration=90, seed=None)
        stats = get_workload_stats(df)
        print(f"\nWorkload {i+1}:")
        print(f"  Packets: {stats['num_packets']}")
        print(f"  Active flows: {stats['num_active_flows']}")
        print(f"  Total bytes: {stats['total_bytes'] / 1e9:.2f} GB")
        print(f"  Duration: {stats['duration']:.1f}s")
    
    # Test saving
    df = generate_workload(num_packets=300, duration=90, seed=42)
    save_workload(df, "test_workload.csv")
    print("\nSaved test_workload.csv")
    print("\nFirst 5 rows:")
    print(df.head().to_string())
    
    # Test demand vector
    demand = generate_demand_vector(df)
    print(f"\nDemand vector: {np.count_nonzero(demand)} non-zero flows")
    print(f"Max demand: {demand.max() / 1e9:.2f} GB")
    print(f"Total demand: {demand.sum() / 1e9:.2f} GB")