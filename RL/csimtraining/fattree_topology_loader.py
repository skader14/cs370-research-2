"""
Fat-Tree Topology Loader and Path Computation

Provides utilities for:
- Loading fat-tree topology from JSON
- Computing all paths between host pairs
- Path diversity analysis
- Static feature computation for RL
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import networkx as nx


# Fat-tree k=4 constants
K = 4
NUM_PODS = K
NUM_HOSTS = 16
NUM_FLOWS = NUM_HOSTS * (NUM_HOSTS - 1)  # 240


class FatTreeTopology:
    """
    Fat-tree topology representation with path computation.
    """
    
    def __init__(self, physical_topology_path: Optional[str] = None):
        """
        Initialize fat-tree topology.
        
        Args:
            physical_topology_path: Path to JSON topology file (optional)
        """
        self.k = K
        self.num_hosts = NUM_HOSTS
        self.num_flows = NUM_FLOWS
        
        # Build network graph
        self.graph = nx.Graph()
        self._build_topology()
        
        # Load from file if provided (for link capacities, latencies)
        if physical_topology_path:
            self._load_from_json(physical_topology_path)
        
        # Precompute paths and features
        self._compute_all_paths()
        self._compute_static_features()
    
    def _build_topology(self):
        """Build the fat-tree graph structure."""
        k = self.k
        
        # Add core switches
        for i in range(k * k // 4):
            self.graph.add_node(f"core_{i}", type="core")
        
        # Add pods
        for pod in range(k):
            # Aggregation switches
            for agg in range(k // 2):
                agg_name = f"agg_{pod}_{agg}"
                self.graph.add_node(agg_name, type="aggregation", pod=pod)
                
                # Connect to core switches
                # agg_i connects to core switches (k/2)*i to (k/2)*i + k/2 - 1
                for core_idx in range(k // 2):
                    core_name = f"core_{agg * (k // 2) + core_idx}"
                    self.graph.add_edge(agg_name, core_name, 
                                       bw=1e9, latency=0.001)
            
            # Edge switches
            for edge in range(k // 2):
                edge_name = f"edge_{pod}_{edge}"
                self.graph.add_node(edge_name, type="edge", pod=pod)
                
                # Connect to all aggregation switches in same pod
                for agg in range(k // 2):
                    agg_name = f"agg_{pod}_{agg}"
                    self.graph.add_edge(edge_name, agg_name,
                                       bw=1e9, latency=0.001)
                
                # Connect hosts
                for h in range(k // 2):
                    host_idx = pod * (k * k // 4) + edge * (k // 2) + h
                    host_name = f"host_{host_idx}"
                    self.graph.add_node(host_name, type="host", 
                                       pod=pod, edge=edge)
                    self.graph.add_edge(edge_name, host_name,
                                       bw=1e9, latency=0.001)
    
    def _load_from_json(self, path: str):
        """Load link properties from JSON topology file."""
        with open(path) as f:
            data = json.load(f)
        
        # Update edge properties from JSON
        for link in data.get('links', []):
            src = link['source']
            dst = link['destination']
            if self.graph.has_edge(src, dst):
                self.graph[src][dst]['bw'] = link.get('bw', 1e9)
                self.graph[src][dst]['latency'] = link.get('latency', 0.001)
    
    def _compute_all_paths(self):
        """Compute all paths between host pairs."""
        self.paths = {}  # (src, dst) -> list of paths
        self.shortest_path_length = {}
        
        hosts = [n for n in self.graph.nodes() if 'host' in n]
        
        for src in hosts:
            for dst in hosts:
                if src == dst:
                    continue
                
                src_id = int(src.split('_')[1])
                dst_id = int(dst.split('_')[1])
                
                # Find all shortest paths
                try:
                    length = nx.shortest_path_length(self.graph, src, dst)
                    all_paths = list(nx.all_shortest_paths(self.graph, src, dst))
                    self.paths[(src_id, dst_id)] = all_paths
                    self.shortest_path_length[(src_id, dst_id)] = length
                except nx.NetworkXNoPath:
                    self.paths[(src_id, dst_id)] = []
                    self.shortest_path_length[(src_id, dst_id)] = float('inf')
    
    def _compute_static_features(self):
        """Compute static features for all flows."""
        self.static_features = {}
        
        for flow_id in range(self.num_flows):
            src, dst = self.flow_id_to_nodes(flow_id)
            
            paths = self.paths.get((src, dst), [])
            path_length = self.shortest_path_length.get((src, dst), 0)
            
            # Compute propagation delay (sum of link latencies)
            if paths:
                path = paths[0]  # Use first path for delay calculation
                prop_delay = 0
                for i in range(len(path) - 1):
                    edge_data = self.graph.get_edge_data(path[i], path[i+1])
                    prop_delay += edge_data.get('latency', 0.001)
            else:
                prop_delay = 0
            
            # Compute bottleneck capacity
            if paths:
                path = paths[0]
                min_bw = float('inf')
                for i in range(len(path) - 1):
                    edge_data = self.graph.get_edge_data(path[i], path[i+1])
                    min_bw = min(min_bw, edge_data.get('bw', 1e9))
                bottleneck = min_bw
            else:
                bottleneck = 0
            
            self.static_features[flow_id] = {
                'propagation_delay': prop_delay,
                'path_length': path_length,
                'bottleneck_capacity': bottleneck,
                'num_paths': len(paths),
                'flow_type': self._get_flow_type(src, dst),
            }
    
    def _get_flow_type(self, src: int, dst: int) -> str:
        """Classify flow type."""
        src_pod = src // (self.k * self.k // 4)
        dst_pod = dst // (self.k * self.k // 4)
        
        if src_pod != dst_pod:
            return 'inter_pod'
        
        src_edge = (src % (self.k * self.k // 4)) // (self.k // 2)
        dst_edge = (dst % (self.k * self.k // 4)) // (self.k // 2)
        
        if src_edge == dst_edge:
            return 'intra_edge'
        else:
            return 'intra_pod'
    
    def flow_id_to_nodes(self, flow_id: int) -> Tuple[int, int]:
        """Convert flow ID to (src, dst) host pair."""
        src = flow_id // (self.num_hosts - 1)
        dst_idx = flow_id % (self.num_hosts - 1)
        dst = dst_idx if dst_idx < src else dst_idx + 1
        return src, dst
    
    def nodes_to_flow_id(self, src: int, dst: int) -> int:
        """Convert (src, dst) host pair to flow ID."""
        dst_idx = dst if dst < src else dst - 1
        return src * (self.num_hosts - 1) + dst_idx
    
    def get_flow_paths(self, flow_id: int) -> List[List[str]]:
        """Get all shortest paths for a flow."""
        src, dst = self.flow_id_to_nodes(flow_id)
        return self.paths.get((src, dst), [])
    
    def get_static_features(self, flow_id: int) -> Dict:
        """Get static features for a flow."""
        return self.static_features.get(flow_id, {})
    
    def get_links_on_path(self, path: List[str]) -> List[Tuple[str, str]]:
        """Get list of (src, dst) link tuples for a path."""
        return [(path[i], path[i+1]) for i in range(len(path) - 1)]
    
    def get_all_links(self) -> List[Tuple[str, str]]:
        """Get all links in the topology."""
        return list(self.graph.edges())
    
    def analyze_path_diversity(self) -> Dict:
        """Analyze path diversity across all flows."""
        diversity = {'1_path': 0, '2_paths': 0, '4_paths': 0}
        
        for flow_id in range(self.num_flows):
            num_paths = self.static_features[flow_id]['num_paths']
            if num_paths == 1:
                diversity['1_path'] += 1
            elif num_paths == 2:
                diversity['2_paths'] += 1
            elif num_paths >= 4:
                diversity['4_paths'] += 1
        
        return diversity


class FatTreeFeatureExtractor:
    """
    Feature extraction for fat-tree topology.
    
    9 features per flow:
    0: propagation_delay (static)
    1: path_length (static)
    2: bottleneck_capacity (static)
    3: demand (dynamic - current episode)
    4: prev_mean_queuing (historical)
    5: prev_max_queuing (historical)
    6: prev_drop_rate (historical)
    7: prev_path_utilization (historical)
    8: prev_bottleneck_util (historical)
    """
    
    def __init__(self, topology: Optional[FatTreeTopology] = None):
        self.num_flows = NUM_FLOWS
        self.num_features = 9
        
        if topology is None:
            self.topology = FatTreeTopology()
        else:
            self.topology = topology
        
        # Initialize static features
        self._init_static_features()
        
        # Historical features (updated after each episode)
        self.prev_mean_queuing = np.zeros(self.num_flows)
        self.prev_max_queuing = np.zeros(self.num_flows)
        self.prev_drop_rate = np.zeros(self.num_flows)
        self.prev_path_util = np.zeros(self.num_flows)
        self.prev_bottleneck_util = np.zeros(self.num_flows)
        
        # Normalization constants
        self.max_delay = 0.01  # 10ms
        self.max_path_length = 6
        self.max_capacity = 1e9
        self.max_demand = 1e9
        self.max_queuing = 0.1  # 100ms
    
    def _init_static_features(self):
        """Initialize static features from topology."""
        self.propagation_delays = np.zeros(self.num_flows)
        self.path_lengths = np.zeros(self.num_flows)
        self.bottleneck_capacities = np.zeros(self.num_flows)
        
        for flow_id in range(self.num_flows):
            features = self.topology.get_static_features(flow_id)
            self.propagation_delays[flow_id] = features.get('propagation_delay', 0.006)
            self.path_lengths[flow_id] = features.get('path_length', 6)
            self.bottleneck_capacities[flow_id] = features.get('bottleneck_capacity', 1e9)
    
    def extract_features(self, demand_vector: np.ndarray) -> np.ndarray:
        """
        Extract normalized feature vector for policy input.
        
        Args:
            demand_vector: Array of shape (num_flows,) with bytes per flow
        
        Returns:
            Flattened array of shape (num_flows * num_features,)
        """
        features = np.zeros((self.num_flows, self.num_features))
        
        for flow_id in range(self.num_flows):
            # Static features (normalized)
            features[flow_id, 0] = self.propagation_delays[flow_id] / self.max_delay
            features[flow_id, 1] = self.path_lengths[flow_id] / self.max_path_length
            features[flow_id, 2] = self.bottleneck_capacities[flow_id] / self.max_capacity
            
            # Dynamic feature
            features[flow_id, 3] = min(demand_vector[flow_id] / self.max_demand, 1.0)
            
            # Historical features
            features[flow_id, 4] = min(self.prev_mean_queuing[flow_id] / self.max_queuing, 1.0)
            features[flow_id, 5] = min(self.prev_max_queuing[flow_id] / self.max_queuing, 1.0)
            features[flow_id, 6] = self.prev_drop_rate[flow_id]
            features[flow_id, 7] = self.prev_path_util[flow_id]
            features[flow_id, 8] = self.prev_bottleneck_util[flow_id]
        
        return features.flatten()
    
    def update_historical_features(self, flow_stats: Dict):
        """
        Update historical features from episode results.
        
        Args:
            flow_stats: Dict mapping flow_id to stats dict with:
                - mean_queuing, max_queuing, drop_rate, path_util, bottleneck_util
        """
        # Reset to zeros
        self.prev_mean_queuing.fill(0)
        self.prev_max_queuing.fill(0)
        self.prev_drop_rate.fill(0)
        self.prev_path_util.fill(0)
        self.prev_bottleneck_util.fill(0)
        
        # Update with new stats
        for flow_id, stats in flow_stats.items():
            if isinstance(flow_id, str):
                flow_id = int(flow_id)
            if 0 <= flow_id < self.num_flows:
                self.prev_mean_queuing[flow_id] = stats.get('mean_queuing', 0)
                self.prev_max_queuing[flow_id] = stats.get('max_queuing', 0)
                self.prev_drop_rate[flow_id] = stats.get('drop_rate', 0)
                self.prev_path_util[flow_id] = stats.get('path_util', 0)
                self.prev_bottleneck_util[flow_id] = stats.get('bottleneck_util', 0)


def load_topology(topology_path: str = None) -> FatTreeTopology:
    """Load fat-tree topology from file or create default."""
    return FatTreeTopology(topology_path)


if __name__ == "__main__":
    print("Fat-Tree Topology Loader Test")
    print("=" * 50)
    
    # Create topology
    topo = FatTreeTopology()
    
    print(f"Hosts: {topo.num_hosts}")
    print(f"Flows: {topo.num_flows}")
    print(f"Links: {len(topo.get_all_links())}")
    
    # Analyze path diversity
    diversity = topo.analyze_path_diversity()
    print(f"\nPath diversity:")
    for paths, count in diversity.items():
        print(f"  {paths}: {count} flows ({count/topo.num_flows*100:.1f}%)")
    
    # Sample flow analysis
    print("\nSample flows:")
    for flow_id in [0, 50, 100, 150, 200]:
        src, dst = topo.flow_id_to_nodes(flow_id)
        features = topo.get_static_features(flow_id)
        paths = topo.get_flow_paths(flow_id)
        print(f"  Flow {flow_id} (host_{src} → host_{dst}):")
        print(f"    Type: {features['flow_type']}")
        print(f"    Paths: {features['num_paths']}")
        print(f"    Hops: {features['path_length']}")
    
    # Test feature extractor
    print("\nFeature Extractor Test:")
    extractor = FatTreeFeatureExtractor(topo)
    demand = np.random.uniform(0, 1e8, size=NUM_FLOWS)
    features = extractor.extract_features(demand)
    print(f"  Feature vector shape: {features.shape}")
    print(f"  Expected: ({NUM_FLOWS * 9},)")