"""
feature_extractor.py - Compute 9 features per flow for policy network.

Features (all normalized to [0, 1]):
    Static (3):
        1. propagation_delay - Physical latency of shortest path
        2. path_length - Number of hops
        3. bottleneck_capacity - Minimum link capacity on path
    
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
# Constants
# =============================================================================

NUM_NODES = 12
NUM_FLOWS = NUM_NODES * (NUM_NODES - 1)  # 132
NUM_FEATURES = 9

# Normalization constants (derived from Abilene topology)
MAX_PROPAGATION_DELAY = 0.020    # 20ms max propagation
MAX_PATH_LENGTH = 6              # Max hops in Abilene
MAX_CAPACITY = 10e9              # 10 Gbps
MAX_DEMAND = 2e9                 # 2 GB max reasonable demand per flow
MAX_QUEUING = 10.0               # 10 seconds (cap for normalization)


# =============================================================================
# Abilene Topology Data
# =============================================================================

# Node names in order (index = node ID)
NODE_NAMES = [
    "Seattle", "Sunnyvale", "LosAngeles", "Denver", "KansasCity",
    "Houston", "Chicago", "Indianapolis", "Atlanta", "Washington",
    "NewYork", "Jacksonville"
]

# Static topology data: propagation delays and path lengths
# Pre-computed for Abilene (would load from file in production)
# Format: TOPOLOGY[src][dst] = (propagation_delay_ms, path_length, bottleneck_capacity)

def _init_topology() -> Dict[int, Dict[int, Tuple[float, int, float]]]:
    """
    Initialize static topology data.
    
    In production, this would be loaded from the physical topology JSON.
    For now, we use reasonable estimates based on Abilene.
    """
    topology = {}
    
    # Approximate propagation delays between Abilene nodes (ms)
    # Based on geographic distances
    prop_delays = {
        (0, 1): 3,   # Seattle-Sunnyvale
        (0, 3): 5,   # Seattle-Denver
        (1, 2): 2,   # Sunnyvale-LosAngeles
        (1, 3): 4,   # Sunnyvale-Denver
        (2, 5): 6,   # LosAngeles-Houston
        (3, 4): 3,   # Denver-KansasCity
        (4, 5): 4,   # KansasCity-Houston
        (4, 6): 3,   # KansasCity-Chicago
        (4, 7): 2,   # KansasCity-Indianapolis
        (5, 8): 4,   # Houston-Atlanta
        (5, 11): 4,  # Houston-Jacksonville
        (6, 7): 2,   # Chicago-Indianapolis
        (6, 10): 4,  # Chicago-NewYork
        (7, 8): 3,   # Indianapolis-Atlanta
        (7, 9): 3,   # Indianapolis-Washington
        (8, 9): 3,   # Atlanta-Washington
        (8, 11): 2,  # Atlanta-Jacksonville
        (9, 10): 2,  # Washington-NewYork
    }
    
    # Make symmetric
    for (i, j), delay in list(prop_delays.items()):
        prop_delays[(j, i)] = delay
    
    # Compute shortest paths using Floyd-Warshall-like approach
    # Initialize with direct links
    dist = np.full((NUM_NODES, NUM_NODES), np.inf)
    hops = np.full((NUM_NODES, NUM_NODES), NUM_NODES + 1)
    
    for i in range(NUM_NODES):
        dist[i, i] = 0
        hops[i, i] = 0
    
    for (i, j), delay in prop_delays.items():
        dist[i, j] = delay
        hops[i, j] = 1
    
    # Floyd-Warshall
    for k in range(NUM_NODES):
        for i in range(NUM_NODES):
            for j in range(NUM_NODES):
                if dist[i, k] + dist[k, j] < dist[i, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]
                    hops[i, j] = hops[i, k] + hops[k, j]
    
    # Build topology dict
    for src in range(NUM_NODES):
        topology[src] = {}
        for dst in range(NUM_NODES):
            if src != dst:
                prop_delay = dist[src, dst] / 1000.0  # Convert to seconds
                path_len = int(hops[src, dst])
                capacity = 10e9  # 10 Gbps (uniform in Abilene)
                topology[src][dst] = (prop_delay, path_len, capacity)
    
    return topology


TOPOLOGY = _init_topology()


# =============================================================================
# Helper Functions
# =============================================================================

def flow_id_to_nodes(flow_id: int) -> Tuple[int, int]:
    """Convert flow ID to (src, dst) node indices."""
    src = flow_id // (NUM_NODES - 1)
    dst_idx = flow_id % (NUM_NODES - 1)
    dst = dst_idx if dst_idx < src else dst_idx + 1
    return src, dst


def nodes_to_flow_id(src: int, dst: int) -> int:
    """Convert (src, dst) to flow ID."""
    dst_idx = dst if dst < src else dst - 1
    return src * (NUM_NODES - 1) + dst_idx


def get_path_nodes(src: int, dst: int) -> List[str]:
    """
    Get node names on the path from src to dst.
    
    This is a simplified version - in production would use actual routing.
    For now, returns just src and dst switch names.
    """
    src_name = f"Switch: {NODE_NAMES[src]}"
    dst_name = f"Switch: {NODE_NAMES[dst]}"
    return [src_name, dst_name]


# =============================================================================
# Episode History Tracker
# =============================================================================

@dataclass
class EpisodeHistory:
    """Stores statistics from the previous episode for feature computation."""
    
    # Per-flow statistics
    flow_mean_queuing: np.ndarray = field(default_factory=lambda: np.zeros(NUM_FLOWS))
    flow_max_queuing: np.ndarray = field(default_factory=lambda: np.zeros(NUM_FLOWS))
    flow_drop_rate: np.ndarray = field(default_factory=lambda: np.zeros(NUM_FLOWS))
    
    # Per-link statistics (keyed by link_id string)
    link_avg_util: Dict[str, float] = field(default_factory=dict)
    link_max_util: Dict[str, float] = field(default_factory=dict)
    
    # Global stats
    global_mean_queuing: float = 0.0
    global_max_queuing: float = 0.0
    global_drop_rate: float = 0.0
    
    def update_from_episode(
        self,
        flow_summary: pd.DataFrame,
        link_stats: pd.DataFrame,
        episode_summary: Dict[str, Any]
    ) -> None:
        """Update history from episode results."""
        
        # Update per-flow stats
        self.flow_mean_queuing.fill(0)
        self.flow_max_queuing.fill(0)
        self.flow_drop_rate.fill(0)
        
        if len(flow_summary) > 0:
            for _, row in flow_summary.iterrows():
                flow_id = int(row['flow_id'])
                if 0 <= flow_id < NUM_FLOWS:
                    self.flow_mean_queuing[flow_id] = row.get('mean_queuing', 0)
                    self.flow_max_queuing[flow_id] = row.get('max_queuing', 0)
                    self.flow_drop_rate[flow_id] = row.get('drop_rate', 0)
        
        # Update per-link stats
        self.link_avg_util.clear()
        self.link_max_util.clear()
        
        if len(link_stats) > 0:
            for _, row in link_stats.iterrows():
                link_id = row['link_id']
                self.link_avg_util[link_id] = row.get('avg_utilization', 0)
                self.link_max_util[link_id] = row.get('max_utilization', 0)
        
        # Update global stats
        self.global_mean_queuing = episode_summary.get('mean_queuing_ms', 0) / 1000.0
        self.global_max_queuing = episode_summary.get('max_queuing_ms', 0) / 1000.0
        self.global_drop_rate = episode_summary.get('drop_rate', 0)
    
    def get_path_utilization(self, src: int, dst: int) -> Tuple[float, float]:
        """
        Get (avg_util, max_util) for the path from src to dst.
        
        Searches for links that include the src or dst switches.
        """
        src_name = NODE_NAMES[src]
        dst_name = NODE_NAMES[dst]
        
        avg_utils = []
        max_utils = []
        
        for link_id, avg_util in self.link_avg_util.items():
            # Check if this link is relevant to the path
            if src_name in link_id or dst_name in link_id:
                avg_utils.append(avg_util)
                max_utils.append(self.link_max_util.get(link_id, avg_util))
        
        if not avg_utils:
            return 0.0, 0.0
        
        return np.mean(avg_utils), np.max(max_utils)
    
    def initialize_random(self, rng: np.random.Generator) -> None:
        """Initialize with random values for cold start."""
        self.flow_mean_queuing = rng.uniform(0, 0.1, NUM_FLOWS)  # 0-100ms
        self.flow_max_queuing = rng.uniform(0, 0.3, NUM_FLOWS)   # 0-300ms
        self.flow_drop_rate = rng.uniform(0, 0.05, NUM_FLOWS)    # 0-5%
        
        self.global_mean_queuing = rng.uniform(0, 0.05)
        self.global_max_queuing = rng.uniform(0, 0.2)
        self.global_drop_rate = rng.uniform(0, 0.02)
        
        # Random link utilizations
        for i in range(NUM_NODES):
            for j in range(NUM_NODES):
                if i != j:
                    link_id = f"Switch: {NODE_NAMES[i]}->Switch: {NODE_NAMES[j]}"
                    self.link_avg_util[link_id] = rng.uniform(0.1, 0.4)
                    self.link_max_util[link_id] = rng.uniform(0.2, 0.6)


# =============================================================================
# Feature Extractor
# =============================================================================

class FeatureExtractor:
    """
    Extracts 9 features per flow for the policy network.
    
    Usage:
        extractor = FeatureExtractor()
        
        # First episode (cold start)
        features = extractor.extract_features(demand_vector)
        
        # After episode completes
        extractor.update_history(flow_summary, link_stats, episode_summary)
        
        # Next episode
        features = extractor.extract_features(new_demand_vector)
    """
    
    def __init__(self, random_cold_start: bool = True, seed: Optional[int] = None):
        """
        Initialize feature extractor.
        
        Args:
            random_cold_start: Use random values for first episode (vs zeros)
            seed: Random seed for cold start
        """
        self.history = EpisodeHistory()
        self.episode_count = 0
        self.rng = np.random.default_rng(seed)
        
        if random_cold_start:
            self.history.initialize_random(self.rng)
    
    def extract_features(self, demand: np.ndarray) -> np.ndarray:
        """
        Extract features for all flows.
        
        Args:
            demand: Array of shape (132,) with bytes per flow
        
        Returns:
            Array of shape (132, 9) with normalized features
        """
        features = np.zeros((NUM_FLOWS, NUM_FEATURES), dtype=np.float32)
        
        # Normalize demand
        max_demand = max(demand.max(), 1.0)  # Avoid division by zero
        
        for flow_id in range(NUM_FLOWS):
            src, dst = flow_id_to_nodes(flow_id)
            prop_delay, path_len, capacity = TOPOLOGY[src][dst]
            
            # === STATIC FEATURES (3) ===
            features[flow_id, 0] = min(1.0, prop_delay / MAX_PROPAGATION_DELAY)
            features[flow_id, 1] = min(1.0, path_len / MAX_PATH_LENGTH)
            features[flow_id, 2] = min(1.0, capacity / MAX_CAPACITY)
            
            # === DYNAMIC FEATURES (1) ===
            features[flow_id, 3] = min(1.0, demand[flow_id] / MAX_DEMAND)
            
            # === HISTORICAL FEATURES (5) ===
            features[flow_id, 4] = min(1.0, self.history.flow_mean_queuing[flow_id] / MAX_QUEUING)
            features[flow_id, 5] = min(1.0, self.history.flow_max_queuing[flow_id] / MAX_QUEUING)
            features[flow_id, 6] = min(1.0, self.history.flow_drop_rate[flow_id])
            
            # Path utilization
            avg_util, max_util = self.history.get_path_utilization(src, dst)
            features[flow_id, 7] = min(1.0, avg_util)
            features[flow_id, 8] = min(1.0, max_util)
        
        return features
    
    def update_history(
        self,
        flow_summary: pd.DataFrame,
        link_stats: pd.DataFrame,
        episode_summary: Dict[str, Any]
    ) -> None:
        """Update history after an episode completes."""
        self.history.update_from_episode(flow_summary, link_stats, episode_summary)
        self.episode_count += 1
    
    def get_feature_names(self) -> List[str]:
        """Get names of the 9 features."""
        return [
            "propagation_delay",
            "path_length", 
            "bottleneck_capacity",
            "demand",
            "prev_mean_queuing",
            "prev_max_queuing",
            "prev_drop_rate",
            "prev_path_utilization",
            "prev_bottleneck_util",
        ]


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("Testing FeatureExtractor")
    print("=" * 50)
    
    # Test topology
    print("\nTopology sample:")
    for flow_id in [0, 50, 84, 131]:
        src, dst = flow_id_to_nodes(flow_id)
        prop, hops, cap = TOPOLOGY[src][dst]
        print(f"  Flow {flow_id}: {NODE_NAMES[src]} -> {NODE_NAMES[dst]}")
        print(f"    prop={prop*1000:.1f}ms, hops={hops}, cap={cap/1e9:.0f}Gbps")
    
    # Test feature extraction
    extractor = FeatureExtractor(random_cold_start=True, seed=42)
    
    # Create dummy demand
    demand = np.zeros(NUM_FLOWS)
    demand[84] = 500_000_000  # 500 MB
    demand[50] = 300_000_000  # 300 MB
    demand[109] = 200_000_000  # 200 MB
    
    features = extractor.extract_features(demand)
    
    print("\nFeature extraction (cold start):")
    print(f"  Shape: {features.shape}")
    print(f"  Feature names: {extractor.get_feature_names()}")
    
    print("\nSample features for high-demand flows:")
    for flow_id in [84, 50, 109, 0]:
        src, dst = flow_id_to_nodes(flow_id)
        print(f"\n  Flow {flow_id} ({NODE_NAMES[src]} -> {NODE_NAMES[dst]}):")
        for i, name in enumerate(extractor.get_feature_names()):
            print(f"    {name}: {features[flow_id, i]:.4f}")
    
    # Test history update
    print("\nSimulating history update...")
    fake_flow_summary = pd.DataFrame({
        'flow_id': [84, 50, 109],
        'mean_queuing': [0.5, 0.3, 0.1],
        'max_queuing': [2.0, 1.5, 0.5],
        'drop_rate': [0.01, 0.005, 0.0],
    })
    fake_link_stats = pd.DataFrame({
        'link_id': ['Switch: Denver->Switch: KansasCity', 'Switch: Atlanta->Switch: Washington'],
        'avg_utilization': [0.3, 0.2],
        'max_utilization': [0.7, 0.4],
    })
    fake_episode_summary = {
        'mean_queuing_ms': 100,
        'max_queuing_ms': 500,
        'drop_rate': 0.005,
    }
    
    extractor.update_history(fake_flow_summary, fake_link_stats, fake_episode_summary)
    
    # Extract again
    features_after = extractor.extract_features(demand)
    
    print("\nFeatures after history update (flow 84):")
    for i, name in enumerate(extractor.get_feature_names()):
        print(f"  {name}: {features[84, i]:.4f} -> {features_after[84, i]:.4f}")