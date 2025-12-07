"""
workload_generator.py - Generate random workloads for CloudSim training episodes.

Updated for Fat-Tree k=4 topology (16 hosts, 240 flows).

Fat-tree properties:
- Full bisection bandwidth
- Multiple paths for ALL inter-pod flows (4 paths)
- Multiple paths for intra-pod flows (2 paths)
- No "bad" flows - all flows can be routed effectively

CloudSim workload format (CSV with header):
    start,source,z,w1,link,dest,psize,w2
    0.4,vm_7,0,1,flow_84,vm_8,146676818,1
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import os


# =============================================================================
# Fat-Tree k=4 Constants
# =============================================================================

K = 4
NUM_PODS = K
NUM_HOSTS_PER_POD = (K // 2) ** 2  # 4
NUM_HOSTS = NUM_PODS * NUM_HOSTS_PER_POD  # 16
NUM_FLOWS = NUM_HOSTS * (NUM_HOSTS - 1)  # 240

NUM_CORE_SWITCHES = (K // 2) ** 2  # 4
NUM_AGG_PER_POD = K // 2  # 2
NUM_EDGE_PER_POD = K // 2  # 2


# =============================================================================
# Topology Utilities
# =============================================================================

def get_host_pod(host_id: int) -> int:
    """Get the pod number for a host (0-3)."""
    return host_id // NUM_HOSTS_PER_POD


def get_host_edge(host_id: int) -> int:
    """Get the edge switch index within pod for a host (0-1)."""
    return (host_id % NUM_HOSTS_PER_POD) // (K // 2)


def get_flow_type(src: int, dst: int) -> str:
    """
    Classify a flow by its path characteristics.
    
    Returns:
        'intra_edge': Same edge switch (1 path, 2 hops)
        'intra_pod': Same pod, different edge (2 paths, 4 hops)
        'inter_pod': Different pods (4 paths, 6 hops)
    """
    src_pod = get_host_pod(src)
    dst_pod = get_host_pod(dst)
    
    if src_pod != dst_pod:
        return 'inter_pod'
    
    src_edge = get_host_edge(src)
    dst_edge = get_host_edge(dst)
    
    if src_edge == dst_edge:
        return 'intra_edge'
    else:
        return 'intra_pod'


def get_num_paths(src: int, dst: int) -> int:
    """Get the number of equal-cost paths for a flow."""
    flow_type = get_flow_type(src, dst)
    if flow_type == 'intra_edge':
        return 1
    elif flow_type == 'intra_pod':
        return 2
    else:
        return 4


def get_path_length(src: int, dst: int) -> int:
    """Get the path length in hops for a flow."""
    flow_type = get_flow_type(src, dst)
    if flow_type == 'intra_edge':
        return 2
    elif flow_type == 'intra_pod':
        return 4
    else:
        return 6


def flow_id_to_nodes(flow_id: int) -> Tuple[int, int]:
    """Convert flow ID to (src, dst) host pair."""
    src = flow_id // (NUM_HOSTS - 1)
    dst_idx = flow_id % (NUM_HOSTS - 1)
    dst = dst_idx if dst_idx < src else dst_idx + 1
    return src, dst


def nodes_to_flow_id(src: int, dst: int) -> int:
    """Convert (src, dst) host pair to flow ID."""
    dst_idx = dst if dst < src else dst - 1
    return src * (NUM_HOSTS - 1) + dst_idx


# =============================================================================
# Pre-computed Flow Classifications
# =============================================================================

def _classify_all_flows() -> Dict[str, List[int]]:
    """Pre-compute flow classifications."""
    flows = {'intra_edge': [], 'intra_pod': [], 'inter_pod': []}
    
    for flow_id in range(NUM_FLOWS):
        src, dst = flow_id_to_nodes(flow_id)
        flow_type = get_flow_type(src, dst)
        flows[flow_type].append(flow_id)
    
    return flows


# Pre-compute at module load
FLOWS_BY_TYPE = _classify_all_flows()
INTER_POD_FLOWS = FLOWS_BY_TYPE['inter_pod']    # 192 flows with 4 paths each
INTRA_POD_FLOWS = FLOWS_BY_TYPE['intra_pod']    # 32 flows with 2 paths each
INTRA_EDGE_FLOWS = FLOWS_BY_TYPE['intra_edge']  # 16 flows with 1 path each
MULTI_PATH_FLOWS = INTER_POD_FLOWS + INTRA_POD_FLOWS  # 224 flows with 2+ paths


# =============================================================================
# Workload Generation
# =============================================================================

def generate_workload(
    num_packets: int = 60,
    duration: float = 20.0,
    seed: Optional[int] = None,
    min_packet_size: int = 10_000,          # 10 KB
    max_packet_size: int = 500_000,         # 500 KB
    flow_selection: str = 'balanced',        # 'balanced', 'inter_pod', 'all'
    flow_pool: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Generate a random workload for one training episode.
    
    Flow selection strategies:
    - 'balanced': Mix of inter-pod (60%), intra-pod (30%), intra-edge (10%)
    - 'inter_pod': Only inter-pod flows (max path diversity, 4 paths each)
    - 'intra_pod': Only intra-pod flows (2 paths each)
    - 'multi_path': Only flows with 2+ paths (excludes intra-edge)
    - 'all': Uniform random from all 240 flows
    
    Args:
        num_packets: Number of packets (= number of active flows)
        duration: Simulation duration in seconds
        seed: Random seed
        min_packet_size: Minimum packet size in bytes
        max_packet_size: Maximum packet size in bytes
        flow_selection: Strategy for selecting which flows are active
        flow_pool: Custom list of flow IDs (overrides flow_selection)
    
    Returns:
        DataFrame in CloudSim format
    """
    rng = np.random.default_rng(seed)
    
    # Determine flow pool
    if flow_pool is not None:
        available_flows = flow_pool
        selected_flows = rng.choice(available_flows, size=min(num_packets, len(available_flows)), replace=False).tolist()
    
    elif flow_selection == 'balanced':
        # Weighted selection: 60% inter-pod, 30% intra-pod, 10% intra-edge
        n_inter = int(num_packets * 0.6)
        n_intra_pod = int(num_packets * 0.3)
        n_intra_edge = num_packets - n_inter - n_intra_pod
        
        selected_flows = []
        if n_inter > 0 and len(INTER_POD_FLOWS) >= n_inter:
            selected_flows.extend(rng.choice(INTER_POD_FLOWS, size=n_inter, replace=False).tolist())
        if n_intra_pod > 0 and len(INTRA_POD_FLOWS) >= n_intra_pod:
            selected_flows.extend(rng.choice(INTRA_POD_FLOWS, size=n_intra_pod, replace=False).tolist())
        if n_intra_edge > 0 and len(INTRA_EDGE_FLOWS) >= n_intra_edge:
            selected_flows.extend(rng.choice(INTRA_EDGE_FLOWS, size=n_intra_edge, replace=False).tolist())
    
    elif flow_selection == 'inter_pod':
        num_active = min(num_packets, len(INTER_POD_FLOWS))
        selected_flows = rng.choice(INTER_POD_FLOWS, size=num_active, replace=False).tolist()
    
    elif flow_selection == 'intra_pod':
        num_active = min(num_packets, len(INTRA_POD_FLOWS))
        selected_flows = rng.choice(INTRA_POD_FLOWS, size=num_active, replace=False).tolist()
    
    elif flow_selection == 'multi_path':
        num_active = min(num_packets, len(MULTI_PATH_FLOWS))
        selected_flows = rng.choice(MULTI_PATH_FLOWS, size=num_active, replace=False).tolist()
    
    else:  # 'all'
        num_active = min(num_packets, NUM_FLOWS)
        selected_flows = rng.choice(NUM_FLOWS, size=num_active, replace=False).tolist()
    
    # Generate packets
    packets = []
    for flow_id in selected_flows:
        src, dst = flow_id_to_nodes(flow_id)
        
        # Random start time within duration (leave 5% buffer at end)
        start_time = rng.uniform(0, duration * 0.95)
        
        # Log-uniform packet size (favors smaller sizes)
        log_min = np.log(min_packet_size)
        log_max = np.log(max_packet_size)
        packet_size = int(np.exp(rng.uniform(log_min, log_max)))
        
        packets.append({
            'start': round(start_time, 6),
            'source': f'vm_{src}',
            'z': 0,
            'w1': 1,
            'link': f'flow_{flow_id}',
            'dest': f'vm_{dst}',
            'psize': packet_size,
            'w2': 1,
        })
    
    df = pd.DataFrame(packets)
    df = df.sort_values('start').reset_index(drop=True)
    return df


def save_workload(df: pd.DataFrame, path: str) -> None:
    """Save workload DataFrame to CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_workload(path: str) -> pd.DataFrame:
    """Load workload from CSV."""
    return pd.read_csv(path)


def generate_demand_vector(workload_df: pd.DataFrame) -> np.ndarray:
    """
    Convert workload DataFrame to demand vector.
    
    Returns:
        Array of shape (NUM_FLOWS,) with total bytes per flow
    """
    demand = np.zeros(NUM_FLOWS, dtype=np.float64)
    
    for _, row in workload_df.iterrows():
        # Extract flow ID from 'link' column (format: 'flow_123')
        link = row.get('link', '')
        if isinstance(link, str) and 'flow_' in link:
            try:
                flow_id = int(link.split('_')[1])
                if 0 <= flow_id < NUM_FLOWS:
                    demand[flow_id] += row.get('psize', 0)
            except (IndexError, ValueError):
                pass
    
    return demand


def get_workload_stats(workload_df: pd.DataFrame) -> Dict:
    """Get statistics about a workload."""
    if workload_df.empty:
        return {'num_packets': 0}
    
    total_bytes = workload_df['psize'].sum()
    duration = workload_df['start'].max() - workload_df['start'].min()
    
    # Analyze flow types
    flow_types = {'intra_edge': 0, 'intra_pod': 0, 'inter_pod': 0}
    for _, row in workload_df.iterrows():
        link = row.get('link', '')
        if isinstance(link, str) and 'flow_' in link:
            try:
                flow_id = int(link.split('_')[1])
                src, dst = flow_id_to_nodes(flow_id)
                flow_type = get_flow_type(src, dst)
                flow_types[flow_type] += 1
            except (IndexError, ValueError):
                pass
    
    return {
        'num_packets': len(workload_df),
        'total_bytes_gb': total_bytes / 1e9,
        'duration_sec': duration,
        'bytes_per_sec_gbps': (total_bytes * 8 / 1e9) / max(duration, 0.001),
        'avg_packet_mb': (total_bytes / len(workload_df)) / 1e6,
        'flow_type_distribution': flow_types,
    }


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_flow_distribution() -> Dict:
    """Analyze the distribution of flow types in fat-tree."""
    return {
        'counts': {
            'inter_pod': len(INTER_POD_FLOWS),
            'intra_pod': len(INTRA_POD_FLOWS),
            'intra_edge': len(INTRA_EDGE_FLOWS),
        },
        'total': NUM_FLOWS,
        'percentages': {
            'inter_pod': len(INTER_POD_FLOWS) / NUM_FLOWS * 100,
            'intra_pod': len(INTRA_POD_FLOWS) / NUM_FLOWS * 100,
            'intra_edge': len(INTRA_EDGE_FLOWS) / NUM_FLOWS * 100,
        }
    }


# =============================================================================
# Main / Testing
# =============================================================================

if __name__ == "__main__":
    print("Fat-Tree k=4 Workload Generator")
    print("=" * 50)
    print(f"Hosts: {NUM_HOSTS}")
    print(f"Total possible flows: {NUM_FLOWS}")
    
    # Flow distribution
    analysis = analyze_flow_distribution()
    print("\nFlow Distribution:")
    for flow_type, count in analysis['counts'].items():
        pct = analysis['percentages'][flow_type]
        paths = {'intra_edge': 1, 'intra_pod': 2, 'inter_pod': 4}[flow_type]
        hops = {'intra_edge': 2, 'intra_pod': 4, 'inter_pod': 6}[flow_type]
        print(f"  {flow_type}: {count} flows ({pct:.1f}%) - {paths} paths, {hops} hops")
    
    # Test workload generation
    print("\nSample workload generation:")
    for strategy in ['balanced', 'inter_pod', 'all']:
        df = generate_workload(num_packets=60, flow_selection=strategy, seed=42)
        stats = get_workload_stats(df)
        print(f"\n  Strategy: {strategy}")
        print(f"    Packets: {stats['num_packets']}")
        print(f"    Total: {stats['total_bytes_gb']:.2f} GB")
        print(f"    Flow types: {stats['flow_type_distribution']}")
    
    # Test save/load
    print("\nTest save/load:")
    df = generate_workload(num_packets=10, seed=123)
    save_workload(df, "/tmp/test_workload.csv")
    loaded = load_workload("/tmp/test_workload.csv")
    print(f"  Saved and loaded {len(loaded)} packets")
    
    # Test demand vector
    demand = generate_demand_vector(df)
    print(f"\nDemand vector:")
    print(f"  Shape: {demand.shape}")
    print(f"  Non-zero flows: {np.count_nonzero(demand)}")
    print(f"  Total demand: {demand.sum() / 1e9:.2f} GB")