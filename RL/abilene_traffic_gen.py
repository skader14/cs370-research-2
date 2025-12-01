"""
Traffic Generator for Abilene Topology

Generates realistic WAN traffic patterns:
- Gravity model (traffic proportional to node importance)
- Hotspot patterns (popular destinations)
- Cross-country elephants (long-haul large flows)
"""

import numpy as np
from typing import Tuple, List
from abilene_topology import AbileneToplogy


class AbileneTrafficGenerator:
    """Generate traffic matrices for Abilene topology."""
    
    def __init__(self, topology: AbileneToplogy, seed: int = None):
        self.topo = topology
        self.rng = np.random.default_rng(seed)
        
        # Node importance weights (major cities get more traffic)
        # Based on rough population/data center density
        self.node_weights = np.array([
            0.8,   # Seattle
            1.0,   # Sunnyvale (Silicon Valley)
            0.9,   # Los Angeles
            0.5,   # Denver
            0.4,   # Kansas City
            0.6,   # Houston
            0.9,   # Chicago
            0.5,   # Indianapolis
            0.7,   # Atlanta
            0.9,   # Washington DC
            1.0,   # New York
            0.3,   # Jacksonville
        ])
        self.node_weights /= self.node_weights.sum()  # Normalize
        
        # Precompute flow categories
        self._categorize_flows()
        
    def _categorize_flows(self):
        """Categorize flows by path count."""
        self.single_path_flows = []
        self.multi_path_flows = []
        
        for flow_idx, (src, dst) in enumerate(self.topo.flow_pairs):
            num_paths = self.topo.get_num_paths(src, dst)
            if num_paths <= 1:
                self.single_path_flows.append(flow_idx)
            else:
                self.multi_path_flows.append(flow_idx)
    
    def generate_gravity(self, 
                        total_traffic: float = 10e9,
                        sparsity: float = 0.5) -> np.ndarray:
        """
        Generate traffic using gravity model.
        
        Traffic between nodes i,j proportional to weight[i] * weight[j].
        """
        tm = np.zeros((self.topo.num_nodes, self.topo.num_nodes))
        
        # Compute gravity weights
        for src in range(self.topo.num_nodes):
            for dst in range(self.topo.num_nodes):
                if src != dst and self.rng.random() < sparsity:
                    gravity = self.node_weights[src] * self.node_weights[dst]
                    tm[src, dst] = gravity
        
        # Scale to total traffic
        if tm.sum() > 0:
            tm = tm / tm.sum() * total_traffic
        
        return tm
    
    def generate_bimodal(self,
                        n_elephants: int = 6,
                        elephant_rate: float = 1.5e9,
                        mouse_rate: float = 100e6,
                        mouse_sparsity: float = 0.4) -> np.ndarray:
        """
        Generate bimodal traffic (elephants + mice).
        
        Elephants placed on longer paths to stress backbone.
        """
        tm = np.zeros((self.topo.num_nodes, self.topo.num_nodes))
        
        # Place elephants on multi-path flows (can be optimized)
        elephant_flows = []
        
        # Prioritize cross-country flows for elephants
        cross_country = []
        for flow_idx in self.multi_path_flows:
            src, dst = self.topo.flow_pairs[flow_idx]
            paths = self.topo.get_all_paths(src, dst)
            if paths and len(paths[0]) >= 4:  # At least 3 hops
                cross_country.append(flow_idx)
        
        # Select elephants from cross-country flows
        if len(cross_country) >= n_elephants:
            elephant_indices = self.rng.choice(cross_country, n_elephants, replace=False)
        else:
            # Fall back to any multi-path flow
            elephant_indices = self.rng.choice(
                self.multi_path_flows, 
                min(n_elephants, len(self.multi_path_flows)), 
                replace=False
            )
        
        for flow_idx in elephant_indices:
            src, dst = self.topo.flow_pairs[flow_idx]
            # Add some variance to elephant sizes
            rate = elephant_rate * self.rng.uniform(0.7, 1.3)
            tm[src, dst] = rate
            elephant_flows.append(flow_idx)
        
        # Place mice
        for flow_idx, (src, dst) in enumerate(self.topo.flow_pairs):
            if tm[src, dst] == 0 and self.rng.random() < mouse_sparsity:
                tm[src, dst] = mouse_rate * self.rng.uniform(0.5, 1.5)
        
        return tm
    
    def generate_hotspot(self,
                        hotspot_nodes: List[int] = None,
                        hotspot_rate: float = 800e6,
                        background_rate: float = 100e6,
                        background_sparsity: float = 0.3) -> np.ndarray:
        """
        Generate traffic with hotspot destinations.
        
        Simulates popular services in specific locations.
        """
        tm = np.zeros((self.topo.num_nodes, self.topo.num_nodes))
        
        # Default hotspots: major data center locations
        if hotspot_nodes is None:
            hotspot_nodes = [1, 10, 6]  # Sunnyvale, New York, Chicago
        
        # Traffic TO hotspots (everyone sends to them)
        for hotspot in hotspot_nodes:
            for src in range(self.topo.num_nodes):
                if src != hotspot:
                    tm[src, hotspot] = hotspot_rate * self.rng.uniform(0.5, 1.5)
        
        # Background traffic
        for src in range(self.topo.num_nodes):
            for dst in range(self.topo.num_nodes):
                if src != dst and tm[src, dst] == 0:
                    if self.rng.random() < background_sparsity:
                        tm[src, dst] = background_rate * self.rng.uniform(0.5, 1.5)
        
        return tm
    
    def generate_realistic(self,
                          n_elephants: int = 5,
                          elephant_rate: float = 1e9,
                          total_mice: float = 5e9) -> np.ndarray:
        """
        Generate realistic mix: gravity-based mice + random elephants.
        """
        # Gravity model for background
        tm = self.generate_gravity(total_traffic=total_mice, sparsity=0.5)
        
        # Add elephants on top
        if len(self.multi_path_flows) >= n_elephants:
            elephant_indices = self.rng.choice(self.multi_path_flows, n_elephants, replace=False)
            for flow_idx in elephant_indices:
                src, dst = self.topo.flow_pairs[flow_idx]
                tm[src, dst] = max(tm[src, dst], elephant_rate * self.rng.uniform(0.8, 1.2))
        
        return tm
    
    def tm_to_flow_vector(self, tm: np.ndarray) -> np.ndarray:
        """Convert traffic matrix to flow vector."""
        flows = np.zeros(self.topo.num_flows)
        for flow_idx, (src, dst) in enumerate(self.topo.flow_pairs):
            flows[flow_idx] = tm[src, dst]
        return flows
    
    def flow_vector_to_tm(self, flows: np.ndarray) -> np.ndarray:
        """Convert flow vector to traffic matrix."""
        tm = np.zeros((self.topo.num_nodes, self.topo.num_nodes))
        for flow_idx, (src, dst) in enumerate(self.topo.flow_pairs):
            tm[src, dst] = flows[flow_idx]
        return tm


def test_traffic_gen():
    """Test traffic generation."""
    print("="*60)
    print("ABILENE TRAFFIC GENERATOR TEST")
    print("="*60)
    
    topo = AbileneToplogy()
    gen = AbileneTrafficGenerator(topo, seed=42)
    
    print(f"\nFlow categories:")
    print(f"  Single-path flows: {len(gen.single_path_flows)}")
    print(f"  Multi-path flows: {len(gen.multi_path_flows)}")
    
    # Test gravity model
    print("\n" + "-"*60)
    print("GRAVITY MODEL")
    print("-"*60)
    tm = gen.generate_gravity(total_traffic=10e9)
    print(f"  Total traffic: {tm.sum()/1e9:.2f} Gbps")
    print(f"  Active flows: {np.sum(tm > 0)}")
    
    # Show top flows
    flows = gen.tm_to_flow_vector(tm)
    top_indices = np.argsort(-flows)[:5]
    print(f"  Top flows:")
    for idx in top_indices:
        src, dst = topo.flow_pairs[idx]
        print(f"    {topo.node_abbrev[src]} -> {topo.node_abbrev[dst]}: {flows[idx]/1e6:.0f} Mbps")
    
    # Test bimodal
    print("\n" + "-"*60)
    print("BIMODAL")
    print("-"*60)
    tm = gen.generate_bimodal(n_elephants=6)
    flows = gen.tm_to_flow_vector(tm)
    print(f"  Total traffic: {tm.sum()/1e9:.2f} Gbps")
    print(f"  Elephant flows (>1 Gbps): {np.sum(flows > 1e9)}")
    
    # Test hotspot
    print("\n" + "-"*60)
    print("HOTSPOT")
    print("-"*60)
    tm = gen.generate_hotspot()
    print(f"  Total traffic: {tm.sum()/1e9:.2f} Gbps")
    
    # Traffic to each hotspot
    for hotspot in [1, 10, 6]:
        incoming = tm[:, hotspot].sum()
        print(f"  Traffic to {topo.node_abbrev[hotspot]}: {incoming/1e9:.2f} Gbps")


if __name__ == "__main__":
    test_traffic_gen()