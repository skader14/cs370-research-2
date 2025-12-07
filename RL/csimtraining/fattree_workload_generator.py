"""
Fat-Tree Workload Generator for CFR-RL CloudSim Training

Fat-tree k=4 topology:
- 16 hosts (4 pods × 4 hosts per pod)
- 20 switches (4 core + 8 aggregation + 8 edge)
- 240 possible flows (16 × 15)
- Full bisection bandwidth
- Multiple equal-cost paths for ALL flows

Flow types by path diversity:
- Intra-edge: 2 hosts under same edge switch (8 pairs) - 1 path, 2 hops
- Intra-pod: Different edge switches, same pod (24 pairs) - 2 paths, 4 hops  
- Inter-pod: Different pods (192 pairs) - 4 paths, 6 hops

Unlike Abilene, ALL flows have viable paths and LP can optimize them.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import json


# Fat-tree k=4 constants
K = 4
NUM_PODS = K
NUM_HOSTS_PER_POD = (K // 2) ** 2  # 4
NUM_HOSTS = NUM_PODS * NUM_HOSTS_PER_POD  # 16
NUM_FLOWS = NUM_HOSTS * (NUM_HOSTS - 1)  # 240

NUM_CORE_SWITCHES = (K // 2) ** 2  # 4
NUM_AGG_PER_POD = K // 2  # 2
NUM_EDGE_PER_POD = K // 2  # 2
NUM_AGG_SWITCHES = NUM_PODS * NUM_AGG_PER_POD  # 8
NUM_EDGE_SWITCHES = NUM_PODS * NUM_EDGE_PER_POD  # 8
NUM_SWITCHES = NUM_CORE_SWITCHES + NUM_AGG_SWITCHES + NUM_EDGE_SWITCHES  # 20


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
        return 2  # Through 2 aggregation switches
    else:  # inter_pod
        return 4  # Through 4 core switches


def get_path_length(src: int, dst: int) -> int:
    """Get the path length in hops for a flow."""
    flow_type = get_flow_type(src, dst)
    if flow_type == 'intra_edge':
        return 2  # host → edge → host
    elif flow_type == 'intra_pod':
        return 4  # host → edge → agg → edge → host
    else:  # inter_pod
        return 6  # host → edge → agg → core → agg → edge → host


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


def analyze_flow_distribution() -> Dict:
    """Analyze the distribution of flow types in fat-tree."""
    counts = {'intra_edge': 0, 'intra_pod': 0, 'inter_pod': 0}
    
    for flow_id in range(NUM_FLOWS):
        src, dst = flow_id_to_nodes(flow_id)
        flow_type = get_flow_type(src, dst)
        counts[flow_type] += 1
    
    return {
        'counts': counts,
        'total': NUM_FLOWS,
        'percentages': {k: v / NUM_FLOWS * 100 for k, v in counts.items()}
    }


def get_flows_by_type() -> Dict[str, List[int]]:
    """Get lists of flow IDs grouped by type."""
    flows = {'intra_edge': [], 'intra_pod': [], 'inter_pod': []}
    
    for flow_id in range(NUM_FLOWS):
        src, dst = flow_id_to_nodes(flow_id)
        flow_type = get_flow_type(src, dst)
        flows[flow_type].append(flow_id)
    
    return flows


# Pre-compute flow classifications
FLOWS_BY_TYPE = get_flows_by_type()
INTER_POD_FLOWS = FLOWS_BY_TYPE['inter_pod']  # 192 flows with 4 paths each
INTRA_POD_FLOWS = FLOWS_BY_TYPE['intra_pod']  # 24 flows with 2 paths each
INTRA_EDGE_FLOWS = FLOWS_BY_TYPE['intra_edge']  # 8 flows with 1 path each

# All flows except intra-edge (which have limited routing options)
MULTI_PATH_FLOWS = INTER_POD_FLOWS + INTRA_POD_FLOWS  # 216 flows


def generate_workload(
    num_packets: int = 50,
    duration: float = 20.0,
    seed: Optional[int] = None,
    min_packet_size: int = 20_000_000,      # 20 MB
    max_packet_size: int = 400_000_000,     # 400 MB
    flow_selection: str = 'balanced',        # 'balanced', 'inter_pod', 'all', 'stress'
    flow_pool: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Generate a random workload for one training episode.
    
    Flow selection strategies:
    - 'balanced': Mix of inter-pod (60%), intra-pod (30%), intra-edge (10%)
    - 'inter_pod': Only inter-pod flows (max path diversity, 4 paths each)
    - 'all': Uniform random from all 240 flows
    - 'stress': High concentration to test congestion handling
    - 'multi_path': Only flows with 2+ paths (excludes intra-edge)
    
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
    elif flow_selection == 'inter_pod':
        available_flows = INTER_POD_FLOWS
    elif flow_selection == 'intra_pod':
        available_flows = INTRA_POD_FLOWS
    elif flow_selection == 'multi_path':
        available_flows = MULTI_PATH_FLOWS
    elif flow_selection == 'balanced':
        # Weighted selection: 60% inter-pod, 30% intra-pod, 10% intra-edge
        n_inter = int(num_packets * 0.6)
        n_intra_pod = int(num_packets * 0.3)
        n_intra_edge = num_packets - n_inter - n_intra_pod
        
        selected = []
        if n_inter > 0 and len(INTER_POD_FLOWS) >= n_inter:
            selected.extend(rng.choice(INTER_POD_FLOWS, size=n_inter, replace=False).tolist())
        if n_intra_pod > 0 and len(INTRA_POD_FLOWS) >= n_intra_pod:
            selected.extend(rng.choice(INTRA_POD_FLOWS, size=n_intra_pod, replace=False).tolist())
        if n_intra_edge > 0 and len(INTRA_EDGE_FLOWS) >= n_intra_edge:
            selected.extend(rng.choice(INTRA_EDGE_FLOWS, size=n_intra_edge, replace=False).tolist())
        
        available_flows = selected if selected else list(range(NUM_FLOWS))
    elif flow_selection == 'stress':
        # Concentrate traffic on specific pods to create congestion
        # All flows going TO pod 0 (creates bottleneck at pod 0's aggregation)
        stress_flows = [f for f in range(NUM_FLOWS) 
                       if get_host_pod(flow_id_to_nodes(f)[1]) == 0
                       and get_host_pod(flow_id_to_nodes(f)[0]) != 0]
        available_flows = stress_flows
    else:  # 'all'
        available_flows = list(range(NUM_FLOWS))
    
    # Select active flows
    num_active = min(num_packets, len(available_flows))
    if flow_selection == 'balanced':
        active_flows = available_flows[:num_active]  # Already selected above
    else:
        active_flows = rng.choice(available_flows, size=num_active, replace=False).tolist()
    
    # Generate packets
    packets = []
    for flow_id in active_flows:
        src, dst = flow_id_to_nodes(flow_id)
        
        # Random start time within duration
        start_time = rng.uniform(0, duration * 0.95)
        
        # Log-uniform packet size
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
        flow_id = int(row['link'].split('_')[1])
        demand[flow_id] += row['psize']
    
    return demand


def get_workload_stats(workload_df: pd.DataFrame) -> Dict:
    """Get statistics about a workload."""
    total_bytes = workload_df['psize'].sum()
    duration = workload_df['start'].max() - workload_df['start'].min()
    
    # Analyze flow types
    flow_types = {'intra_edge': 0, 'intra_pod': 0, 'inter_pod': 0}
    for _, row in workload_df.iterrows():
        flow_id = int(row['link'].split('_')[1])
        src, dst = flow_id_to_nodes(flow_id)
        flow_type = get_flow_type(src, dst)
        flow_types[flow_type] += 1
    
    return {
        'num_packets': len(workload_df),
        'total_bytes_gb': total_bytes / 1e9,
        'duration_sec': duration,
        'bytes_per_sec_gbps': (total_bytes * 8 / 1e9) / max(duration, 1),
        'avg_packet_mb': (total_bytes / len(workload_df)) / 1e6 if len(workload_df) > 0 else 0,
        'flow_type_distribution': flow_types,
    }


# Topology information for feature extraction
def get_static_flow_features() -> Dict[int, Dict]:
    """
    Compute static features for all flows.
    
    Returns dict mapping flow_id to:
    - path_length: Number of hops
    - num_paths: Number of equal-cost paths
    - flow_type: 'intra_edge', 'intra_pod', or 'inter_pod'
    - bottleneck_capacity: Min link capacity on path (all 1 Gbps in our setup)
    - propagation_delay: Sum of link delays (1ms per hop)
    """
    features = {}
    
    for flow_id in range(NUM_FLOWS):
        src, dst = flow_id_to_nodes(flow_id)
        flow_type = get_flow_type(src, dst)
        path_length = get_path_length(src, dst)
        
        features[flow_id] = {
            'path_length': path_length,
            'num_paths': get_num_paths(src, dst),
            'flow_type': flow_type,
            'bottleneck_capacity': 1e9,  # 1 Gbps (all links same capacity)
            'propagation_delay': path_length * 0.001,  # 1ms per hop
        }
    
    return features


if __name__ == "__main__":
    # Print topology analysis
    print("Fat-Tree k=4 Topology Analysis")
    print("=" * 50)
    print(f"Hosts: {NUM_HOSTS}")
    print(f"Switches: {NUM_SWITCHES} (4 core + 8 agg + 8 edge)")
    print(f"Total possible flows: {NUM_FLOWS}")
    print()
    
    analysis = analyze_flow_distribution()
    print("Flow Distribution:")
    for flow_type, count in analysis['counts'].items():
        pct = analysis['percentages'][flow_type]
        paths = {'intra_edge': 1, 'intra_pod': 2, 'inter_pod': 4}[flow_type]
        hops = {'intra_edge': 2, 'intra_pod': 4, 'inter_pod': 6}[flow_type]
        print(f"  {flow_type}: {count} flows ({pct:.1f}%) - {paths} paths, {hops} hops")
    
    print()
    print("Sample workload generation:")
    
    for strategy in ['balanced', 'inter_pod', 'all']:
        df = generate_workload(num_packets=50, flow_selection=strategy, seed=42)
        stats = get_workload_stats(df)
        print(f"\n  Strategy: {strategy}")
        print(f"    Packets: {stats['num_packets']}")
        print(f"    Total: {stats['total_bytes_gb']:.2f} GB")
        print(f"    Flow types: {stats['flow_type_distribution']}")