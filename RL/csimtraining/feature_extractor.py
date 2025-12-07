"""
feature_extractor.py - Compute 9 features per flow for policy network.

Updated for Fat-Tree k=4 topology (16 hosts, 240 flows).

Features (all normalized to [0, 1]):
    Static (3):
        1. propagation_delay - Physical latency of shortest path
        2. path_length - Number of hops (2, 4, or 6)
        3. bottleneck_capacity - Minimum link capacity on path (all 1 Gbps)
    
    Dynamic (1):
        4. demand - Current episode's traffic for this flow
    
    Historical (5) - from previous episode:
        5. prev_mean_queuing - Average queuing delay
        6. prev_max_queuing - Worst-case queuing delay
        7. prev_drop_rate - Packet drop rate
        8. prev_path_utilization - Average link utilization on path
        9. prev_bottleneck_util - Maximum link utilization on path
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import json


# =============================================================================
# Constants (Fat-Tree k=4)
# =============================================================================

K = 4  # Fat-tree parameter
NUM_PODS = K
NUM_HOSTS_PER_POD = (K // 2) ** 2  # 4
NUM_HOSTS = NUM_PODS * NUM_HOSTS_PER_POD  # 16
NUM_FLOWS = NUM_HOSTS * (NUM_HOSTS - 1)  # 240
NUM_FEATURES = 9

# Normalization constants
MAX_PROPAGATION_DELAY = 0.010    # 10ms max propagation (6 hops × 1ms + margin)
MAX_PATH_LENGTH = 6              # Max hops in fat-tree k=4
MAX_CAPACITY = 1e9               # 1 Gbps (all links same)
MAX_DEMAND = 2e9                 # 2 GB max reasonable demand per flow
MAX_QUEUING = 10.0               # 10 seconds (cap for normalization)


# =============================================================================
# Fat-Tree Topology Utilities
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


# =============================================================================
# Static Feature Computation
# =============================================================================

def _compute_static_features() -> Dict[int, Dict[str, float]]:
    """
    Pre-compute static features for all 240 flows.
    
    Returns:
        Dict mapping flow_id to feature dict
    """
    features = {}
    
    for flow_id in range(NUM_FLOWS):
        src, dst = flow_id_to_nodes(flow_id)
        flow_type = get_flow_type(src, dst)
        path_length = get_path_length(src, dst)
        num_paths = get_num_paths(src, dst)
        
        # Propagation delay: 1ms per hop
        prop_delay = path_length * 0.001  # seconds
        
        # Bottleneck capacity: all links are 1 Gbps
        bottleneck = 1e9
        
        features[flow_id] = {
            'propagation_delay': prop_delay,
            'path_length': path_length,
            'bottleneck_capacity': bottleneck,
            'num_paths': num_paths,
            'flow_type': flow_type,
        }
    
    return features


# Pre-compute static features at module load
STATIC_FEATURES = _compute_static_features()


# =============================================================================
# Feature Extractor Class
# =============================================================================

class FeatureExtractor:
    """
    Extract 9 features per flow for policy network input.
    
    Features are organized as a (NUM_FLOWS, NUM_FEATURES) matrix,
    then flattened to (NUM_FLOWS * NUM_FEATURES,) for MLP input.
    """
    
    def __init__(self, random_cold_start: bool = False):
        """
        Initialize feature extractor.
        
        Args:
            random_cold_start: If True, initialize historical features
                              with small random values instead of zeros.
        """
        self.num_flows = NUM_FLOWS
        self.num_features = NUM_FEATURES
        
        # Static features (pre-computed)
        self.static_features = STATIC_FEATURES
        
        # Historical features (updated after each episode)
        self._init_historical_features(random=random_cold_start)
    
    def _init_historical_features(self, random: bool = False) -> None:
        """Initialize historical feature arrays."""
        if random:
            # Small random initialization to break symmetry
            self.prev_mean_queuing = np.random.uniform(0, 0.01, NUM_FLOWS)
            self.prev_max_queuing = np.random.uniform(0, 0.02, NUM_FLOWS)
            self.prev_drop_rate = np.random.uniform(0, 0.01, NUM_FLOWS)
            self.prev_path_util = np.random.uniform(0, 0.1, NUM_FLOWS)
            self.prev_bottleneck_util = np.random.uniform(0, 0.1, NUM_FLOWS)
        else:
            self.prev_mean_queuing = np.zeros(NUM_FLOWS)
            self.prev_max_queuing = np.zeros(NUM_FLOWS)
            self.prev_drop_rate = np.zeros(NUM_FLOWS)
            self.prev_path_util = np.zeros(NUM_FLOWS)
            self.prev_bottleneck_util = np.zeros(NUM_FLOWS)
    
    def extract_features(self, demand_vector: np.ndarray) -> np.ndarray:
        """
        Extract normalized feature vector for all flows.
        
        Args:
            demand_vector: Array of shape (NUM_FLOWS,) with bytes per flow
        
        Returns:
            Flattened array of shape (NUM_FLOWS * NUM_FEATURES,)
        """
        features = np.zeros((NUM_FLOWS, NUM_FEATURES))
        
        for flow_id in range(NUM_FLOWS):
            static = self.static_features[flow_id]
            
            # Feature 0: Propagation delay (normalized)
            features[flow_id, 0] = static['propagation_delay'] / MAX_PROPAGATION_DELAY
            
            # Feature 1: Path length (normalized)
            features[flow_id, 1] = static['path_length'] / MAX_PATH_LENGTH
            
            # Feature 2: Bottleneck capacity (normalized)
            features[flow_id, 2] = static['bottleneck_capacity'] / MAX_CAPACITY
            
            # Feature 3: Demand (normalized, capped at 1)
            features[flow_id, 3] = min(demand_vector[flow_id] / MAX_DEMAND, 1.0)
            
            # Feature 4: Previous mean queuing delay (normalized)
            features[flow_id, 4] = min(self.prev_mean_queuing[flow_id] / MAX_QUEUING, 1.0)
            
            # Feature 5: Previous max queuing delay (normalized)
            features[flow_id, 5] = min(self.prev_max_queuing[flow_id] / MAX_QUEUING, 1.0)
            
            # Feature 6: Previous drop rate (already 0-1)
            features[flow_id, 6] = self.prev_drop_rate[flow_id]
            
            # Feature 7: Previous path utilization (already 0-1)
            features[flow_id, 7] = self.prev_path_util[flow_id]
            
            # Feature 8: Previous bottleneck utilization (already 0-1)
            features[flow_id, 8] = self.prev_bottleneck_util[flow_id]
        
        return features.flatten()
    
    def update_historical_features(self, flow_summary: pd.DataFrame) -> None:
        """
        Update historical features from episode results.
        
        Args:
            flow_summary: DataFrame with columns:
                - flow_id
                - mean_queuing_ms (or mean_queuing)
                - max_queuing_ms (or max_queuing)
                - drop_rate (or dropped, total for computation)
                - plus any utilization columns
        """
        # Reset to zeros first
        self._init_historical_features(random=False)
        
        if flow_summary is None or flow_summary.empty:
            return
        
        # Process each row
        for _, row in flow_summary.iterrows():
            # Get flow ID
            flow_id = row.get('flow_id')
            if flow_id is None:
                # Try extracting from flow name like 'flow_42'
                flow_name = row.get('flow', row.get('name', ''))
                if isinstance(flow_name, str) and 'flow_' in flow_name:
                    try:
                        flow_id = int(flow_name.split('_')[1])
                    except (IndexError, ValueError):
                        continue
                else:
                    continue
            
            flow_id = int(flow_id)
            if not (0 <= flow_id < NUM_FLOWS):
                continue
            
            # Mean queuing (try different column names)
            mean_q = row.get('mean_queuing_ms', row.get('mean_queuing', row.get('avg_queuing_ms', 0)))
            if pd.notna(mean_q):
                # Convert ms to seconds if needed
                if mean_q > 100:  # Likely in ms
                    mean_q = mean_q / 1000.0
                self.prev_mean_queuing[flow_id] = float(mean_q)
            
            # Max queuing
            max_q = row.get('max_queuing_ms', row.get('max_queuing', 0))
            if pd.notna(max_q):
                if max_q > 100:  # Likely in ms
                    max_q = max_q / 1000.0
                self.prev_max_queuing[flow_id] = float(max_q)
            
            # Drop rate
            drop_rate = row.get('drop_rate', 0)
            if pd.notna(drop_rate):
                self.prev_drop_rate[flow_id] = float(drop_rate)
            elif 'dropped' in row and 'total' in row:
                total = row.get('total', 1)
                if total > 0:
                    self.prev_drop_rate[flow_id] = float(row.get('dropped', 0)) / total
            
            # Path utilization (if available)
            path_util = row.get('path_util', row.get('path_utilization', 0))
            if pd.notna(path_util):
                self.prev_path_util[flow_id] = float(path_util)
            
            # Bottleneck utilization (if available)
            bottleneck_util = row.get('bottleneck_util', row.get('bottleneck_utilization', 0))
            if pd.notna(bottleneck_util):
                self.prev_bottleneck_util[flow_id] = float(bottleneck_util)
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return [
            'propagation_delay',
            'path_length',
            'bottleneck_capacity',
            'demand',
            'prev_mean_queuing',
            'prev_max_queuing',
            'prev_drop_rate',
            'prev_path_utilization',
            'prev_bottleneck_util',
        ]
    
    def describe_flow(self, flow_id: int) -> Dict[str, Any]:
        """Get human-readable description of a flow."""
        src, dst = flow_id_to_nodes(flow_id)
        static = self.static_features[flow_id]
        
        return {
            'flow_id': flow_id,
            'src_host': src,
            'dst_host': dst,
            'flow_type': static['flow_type'],
            'path_length': static['path_length'],
            'num_paths': static['num_paths'],
            'propagation_delay_ms': static['propagation_delay'] * 1000,
        }


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_flow_distribution() -> Dict[str, int]:
    """Analyze the distribution of flow types in fat-tree."""
    counts = {'intra_edge': 0, 'intra_pod': 0, 'inter_pod': 0}
    
    for flow_id in range(NUM_FLOWS):
        src, dst = flow_id_to_nodes(flow_id)
        flow_type = get_flow_type(src, dst)
        counts[flow_type] += 1
    
    return counts


def get_flows_by_type() -> Dict[str, List[int]]:
    """Get lists of flow IDs grouped by type."""
    flows = {'intra_edge': [], 'intra_pod': [], 'inter_pod': []}
    
    for flow_id in range(NUM_FLOWS):
        src, dst = flow_id_to_nodes(flow_id)
        flow_type = get_flow_type(src, dst)
        flows[flow_type].append(flow_id)
    
    return flows


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("Fat-Tree Feature Extractor Test")
    print("=" * 50)
    
    print(f"\nTopology: Fat-Tree k={K}")
    print(f"  Hosts: {NUM_HOSTS}")
    print(f"  Flows: {NUM_FLOWS}")
    print(f"  Features per flow: {NUM_FEATURES}")
    print(f"  Total feature vector size: {NUM_FLOWS * NUM_FEATURES}")
    
    # Analyze flow distribution
    dist = analyze_flow_distribution()
    print(f"\nFlow type distribution:")
    for flow_type, count in dist.items():
        pct = count / NUM_FLOWS * 100
        print(f"  {flow_type}: {count} ({pct:.1f}%)")
    
    # Test feature extractor
    extractor = FeatureExtractor(random_cold_start=True)
    
    # Create fake demand vector
    demand = np.zeros(NUM_FLOWS)
    demand[0] = 1e8   # 100 MB for flow 0
    demand[100] = 5e8  # 500 MB for flow 100
    
    features = extractor.extract_features(demand)
    print(f"\nFeature extraction:")
    print(f"  Input shape: ({NUM_FLOWS},)")
    print(f"  Output shape: {features.shape}")
    print(f"  Output reshaped: ({NUM_FLOWS}, {NUM_FEATURES})")
    
    # Show sample flow info
    print("\nSample flows:")
    for flow_id in [0, 50, 100, 150, 200]:
        info = extractor.describe_flow(flow_id)
        print(f"  Flow {flow_id}: host_{info['src_host']} → host_{info['dst_host']} "
              f"({info['flow_type']}, {info['path_length']} hops, {info['num_paths']} paths)")