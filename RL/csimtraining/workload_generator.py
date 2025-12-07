"""
workload_generator.py - Generate workloads for CloudSim training episodes.

Updated for Fat-Tree k=4 topology with HOTSPOT and GRAVITY traffic models.

Key insight: For CFR-RL to learn, we need:
1. Uneven congestion - some paths heavily loaded, others free
2. Path diversity - fat-tree provides 2-4 equal-cost paths
3. Learnable patterns - policy can identify flows that benefit from rerouting

Traffic Models:
- UNIFORM: Random traffic (baseline, not useful for learning)
- HOTSPOT: Heavy traffic between specific host pairs, creates bottlenecks
- GRAVITY: Traffic proportional to host "weight", realistic datacenter pattern
- STRIDE: Deterministic pattern that stresses specific links
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
    """Classify a flow by its path characteristics."""
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


FLOWS_BY_TYPE = _classify_all_flows()
INTER_POD_FLOWS = FLOWS_BY_TYPE['inter_pod']    # 192 flows with 4 paths each
INTRA_POD_FLOWS = FLOWS_BY_TYPE['intra_pod']    # 32 flows with 2 paths each
INTRA_EDGE_FLOWS = FLOWS_BY_TYPE['intra_edge']  # 16 flows with 1 path each
MULTI_PATH_FLOWS = INTER_POD_FLOWS + INTRA_POD_FLOWS  # 224 flows with 2+ paths


# =============================================================================
# Traffic Pattern Generators
# =============================================================================

def generate_hotspot_weights(
    rng: np.random.Generator,
    num_hotspots: int = 3,
    hotspot_intensity: float = 10.0,
) -> np.ndarray:
    """
    Generate flow weights with hotspot pattern.
    
    Creates a few high-traffic "hotspot" flows that will congest specific links,
    while other flows have lower background traffic.
    
    Args:
        rng: Random number generator
        num_hotspots: Number of hotspot host pairs (creates concentrated traffic)
        hotspot_intensity: How much more traffic hotspots generate vs background
    
    Returns:
        Array of shape (NUM_FLOWS,) with relative traffic weights
    """
    weights = np.ones(NUM_FLOWS)
    
    # Select hotspot source-destination pairs (prefer inter-pod for max impact)
    # Hotspots should be inter-pod flows that share core links
    hotspot_flows = rng.choice(INTER_POD_FLOWS, size=min(num_hotspots, len(INTER_POD_FLOWS)), replace=False)
    
    for flow_id in hotspot_flows:
        weights[flow_id] = hotspot_intensity
        
        # Also boost flows that share the same source or destination (creates realistic pattern)
        src, dst = flow_id_to_nodes(flow_id)
        for other_flow in range(NUM_FLOWS):
            other_src, other_dst = flow_id_to_nodes(other_flow)
            if other_src == src or other_dst == dst:
                weights[other_flow] = max(weights[other_flow], hotspot_intensity * 0.5)
    
    return weights / weights.sum()  # Normalize to probability distribution


def generate_gravity_weights(
    rng: np.random.Generator,
    weight_variance: float = 2.0,
) -> np.ndarray:
    """
    Generate flow weights using gravity model.
    
    Traffic between hosts i and j is proportional to: weight[i] * weight[j]
    This creates realistic datacenter traffic where some hosts are busier than others.
    
    Args:
        rng: Random number generator  
        weight_variance: How much host weights vary (higher = more skewed)
    
    Returns:
        Array of shape (NUM_FLOWS,) with relative traffic weights
    """
    # Assign random "importance" weights to each host
    # Use log-normal distribution for realistic skew (some hosts much busier)
    host_weights = rng.lognormal(mean=0, sigma=weight_variance, size=NUM_HOSTS)
    
    # Flow weight = src_weight * dst_weight (gravity model)
    flow_weights = np.zeros(NUM_FLOWS)
    for flow_id in range(NUM_FLOWS):
        src, dst = flow_id_to_nodes(flow_id)
        flow_weights[flow_id] = host_weights[src] * host_weights[dst]
    
    return flow_weights / flow_weights.sum()


def generate_stride_weights(stride: int = 4) -> np.ndarray:
    """
    Generate flow weights using stride pattern.
    
    Host i sends to host (i + stride) mod NUM_HOSTS.
    This is a deterministic pattern that stresses specific links predictably.
    
    Args:
        stride: Offset for destination selection
    
    Returns:
        Array of shape (NUM_FLOWS,) with relative traffic weights (0 or 1)
    """
    weights = np.zeros(NUM_FLOWS)
    
    for src in range(NUM_HOSTS):
        dst = (src + stride) % NUM_HOSTS
        if src != dst:
            flow_id = nodes_to_flow_id(src, dst)
            weights[flow_id] = 1.0
    
    # Normalize
    if weights.sum() > 0:
        weights = weights / weights.sum()
    
    return weights


def generate_skewed_weights(
    rng: np.random.Generator,
    skew_factor: float = 0.8,
) -> np.ndarray:
    """
    Generate flow weights with power-law skew.
    
    A few flows get most of the traffic (realistic for many workloads).
    
    Args:
        rng: Random number generator
        skew_factor: Pareto shape parameter (lower = more skewed)
    
    Returns:
        Array of shape (NUM_FLOWS,) with relative traffic weights
    """
    # Pareto distribution creates heavy tail
    weights = rng.pareto(skew_factor, size=NUM_FLOWS)
    return weights / weights.sum()


# =============================================================================
# Main Workload Generation
# =============================================================================

def generate_workload(
    num_packets: int = 100,
    duration: float = 10.0,
    seed: Optional[int] = None,
    min_packet_size: int = 50_000,       # 50 KB
    max_packet_size: int = 1_000_000,    # 1 MB
    traffic_model: str = 'hotspot',      # 'uniform', 'hotspot', 'gravity', 'stride', 'skewed'
    hotspot_count: int = 4,              # Number of hotspot pairs
    hotspot_intensity: float = 15.0,     # Hotspot traffic multiplier
    gravity_variance: float = 1.5,       # Gravity model variance
    stride: int = 4,                     # Stride pattern offset
) -> pd.DataFrame:
    """
    Generate a random workload for one training episode.
    
    Traffic Models:
    - 'uniform': Random traffic across all flows (baseline)
    - 'hotspot': Concentrated traffic on few host pairs (recommended for learning)
    - 'gravity': Traffic proportional to host weights (realistic)
    - 'stride': Deterministic pattern (useful for debugging)
    - 'skewed': Power-law distribution (realistic web traffic)
    
    Args:
        num_packets: Number of packets to generate
        duration: Simulation duration in seconds
        seed: Random seed for reproducibility
        min_packet_size: Minimum packet size in bytes
        max_packet_size: Maximum packet size in bytes
        traffic_model: Traffic distribution model
        hotspot_count: Number of hotspot pairs (for 'hotspot' model)
        hotspot_intensity: Traffic multiplier for hotspots
        gravity_variance: Variance in host weights (for 'gravity' model)
        stride: Destination offset (for 'stride' model)
    
    Returns:
        DataFrame in CloudSim workload format
    """
    rng = np.random.default_rng(seed)
    
    # Generate flow selection probabilities based on traffic model
    if traffic_model == 'hotspot':
        flow_weights = generate_hotspot_weights(rng, hotspot_count, hotspot_intensity)
    elif traffic_model == 'gravity':
        flow_weights = generate_gravity_weights(rng, gravity_variance)
    elif traffic_model == 'stride':
        flow_weights = generate_stride_weights(stride)
    elif traffic_model == 'skewed':
        flow_weights = generate_skewed_weights(rng)
    else:  # 'uniform'
        flow_weights = np.ones(NUM_FLOWS) / NUM_FLOWS
    
    # Sample flows according to weights (with replacement for realistic traffic)
    selected_flows = rng.choice(
        NUM_FLOWS, 
        size=num_packets, 
        replace=True, 
        p=flow_weights
    )
    
    # Generate packets
    packets = []
    for flow_id in selected_flows:
        src, dst = flow_id_to_nodes(flow_id)
        
        # Poisson-like arrival times (exponential inter-arrival)
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
    """Convert workload DataFrame to demand vector."""
    demand = np.zeros(NUM_FLOWS, dtype=np.float64)
    
    for _, row in workload_df.iterrows():
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
    
    # Count unique flows and packets per flow
    flow_counts = workload_df['link'].value_counts()
    
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
        'unique_flows': len(flow_counts),
        'max_packets_per_flow': flow_counts.max() if len(flow_counts) > 0 else 0,
        'total_bytes_mb': total_bytes / 1e6,
        'duration_sec': duration,
        'rate_mbps': (total_bytes * 8 / 1e6) / max(duration, 0.001),
        'avg_packet_kb': (total_bytes / len(workload_df)) / 1e3,
        'flow_type_distribution': flow_types,
    }


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_traffic_pattern(weights: np.ndarray) -> Dict:
    """Analyze a traffic weight distribution."""
    non_zero = weights[weights > 0]
    
    # Gini coefficient (measure of inequality)
    sorted_weights = np.sort(weights)
    n = len(weights)
    cumulative = np.cumsum(sorted_weights)
    gini = (2 * np.sum((np.arange(1, n+1) * sorted_weights))) / (n * np.sum(sorted_weights)) - (n + 1) / n
    
    return {
        'num_active_flows': len(non_zero),
        'max_weight': weights.max(),
        'top_10_share': np.sort(weights)[-10:].sum(),  # Share of top 10 flows
        'gini_coefficient': gini,  # 0 = equal, 1 = all traffic on one flow
    }


# =============================================================================
# Main / Testing
# =============================================================================

if __name__ == "__main__":
    print("Fat-Tree k=4 Workload Generator v3 (Traffic Models)")
    print("=" * 60)
    
    # Compare traffic models
    models = ['uniform', 'hotspot', 'gravity', 'skewed']
    
    for model in models:
        print(f"\n{model.upper()} Model:")
        print("-" * 40)
        
        # Generate sample workload
        df = generate_workload(
            num_packets=100,
            duration=10.0,
            seed=42,
            traffic_model=model,
        )
        
        stats = get_workload_stats(df)
        print(f"  Packets: {stats['num_packets']}")
        print(f"  Unique flows: {stats['unique_flows']}")
        print(f"  Max packets/flow: {stats['max_packets_per_flow']}")
        print(f"  Total: {stats['total_bytes_mb']:.1f} MB")
        print(f"  Rate: {stats['rate_mbps']:.1f} Mbps")
        print(f"  Flow types: {stats['flow_type_distribution']}")
        
        # Analyze the underlying weight distribution
        rng = np.random.default_rng(42)
        if model == 'hotspot':
            weights = generate_hotspot_weights(rng, 4, 15.0)
        elif model == 'gravity':
            weights = generate_gravity_weights(rng, 1.5)
        elif model == 'skewed':
            weights = generate_skewed_weights(rng)
        else:
            weights = np.ones(NUM_FLOWS) / NUM_FLOWS
        
        analysis = analyze_traffic_pattern(weights)
        print(f"  Top 10 flows share: {analysis['top_10_share']*100:.1f}%")
        print(f"  Gini coefficient: {analysis['gini_coefficient']:.3f}")