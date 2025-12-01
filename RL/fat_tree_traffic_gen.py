"""
Traffic Generator for Fat-Tree Topology

Generates realistic data center traffic patterns:
- Bimodal: elephants (large) + mice (small) flows
- Cross-pod biased: most traffic crosses pod boundaries
"""

import numpy as np
from typing import Tuple, List
from fat_tree_topology import FatTreeTopology


class FatTreeTrafficGenerator:
    """Generate traffic matrices for fat-tree topology."""
    
    def __init__(self, topology: FatTreeTopology, seed: int = None):
        self.topo = topology
        self.num_hosts = topology.num_hosts
        self.rng = np.random.default_rng(seed)
        
        # Precompute flow categories
        self._categorize_flows()
        
    def _categorize_flows(self):
        """Categorize flows by path diversity."""
        self.same_edge_flows = []
        self.same_pod_flows = []
        self.cross_pod_flows = []
        
        for flow_idx, (src, dst) in enumerate(self.topo.flow_pairs):
            ftype = self.topo.get_flow_type(src, dst)
            if ftype == "same_edge":
                self.same_edge_flows.append(flow_idx)
            elif ftype == "same_pod":
                self.same_pod_flows.append(flow_idx)
            else:
                self.cross_pod_flows.append(flow_idx)
                
        # Multi-path flows (can be optimized)
        self.multi_path_flows = self.same_pod_flows + self.cross_pod_flows
        
    def generate_bimodal(self,
                         n_elephants: int = 8,
                         elephant_rate: float = 400e6,
                         mouse_rate: float = 50e6,
                         mouse_sparsity: float = 0.3,
                         cross_pod_elephant_prob: float = 0.9) -> np.ndarray:
        """
        Generate bimodal traffic (elephants + mice).
        
        Args:
            n_elephants: Number of elephant flows
            elephant_rate: Bandwidth per elephant (bps)
            mouse_rate: Bandwidth per mouse (bps)
            mouse_sparsity: Fraction of host pairs with mouse flows
            cross_pod_elephant_prob: Probability elephants are cross-pod
            
        Returns:
            Traffic matrix (num_hosts x num_hosts)
        """
        tm = np.zeros((self.num_hosts, self.num_hosts))
        
        # Place elephants - bias towards cross-pod for optimization opportunity
        elephant_flows = []
        for _ in range(n_elephants):
            if self.rng.random() < cross_pod_elephant_prob and self.cross_pod_flows:
                flow_idx = self.rng.choice(self.cross_pod_flows)
            elif self.same_pod_flows:
                flow_idx = self.rng.choice(self.same_pod_flows)
            else:
                flow_idx = self.rng.choice(len(self.topo.flow_pairs))
            
            # Avoid duplicate elephants
            if flow_idx not in elephant_flows:
                elephant_flows.append(flow_idx)
                src, dst = self.topo.flow_pairs[flow_idx]
                tm[src - self.topo.host_offset, dst - self.topo.host_offset] = elephant_rate
        
        # Place mice
        for flow_idx, (src, dst) in enumerate(self.topo.flow_pairs):
            si, di = src - self.topo.host_offset, dst - self.topo.host_offset
            if tm[si, di] == 0 and self.rng.random() < mouse_sparsity:
                tm[si, di] = mouse_rate
                
        return tm
    
    def generate_uniform(self, 
                        rate: float = 100e6,
                        sparsity: float = 0.5) -> np.ndarray:
        """Generate uniform random traffic."""
        tm = np.zeros((self.num_hosts, self.num_hosts))
        
        for src in self.topo.host_ids:
            for dst in self.topo.host_ids:
                if src != dst and self.rng.random() < sparsity:
                    tm[src - self.topo.host_offset, dst - self.topo.host_offset] = rate
                    
        return tm
    
    def generate_hotspot(self,
                        n_hotspots: int = 2,
                        hotspot_rate: float = 300e6,
                        background_rate: float = 50e6,
                        background_sparsity: float = 0.2) -> np.ndarray:
        """
        Generate traffic with hotspot hosts that send/receive more.
        
        Args:
            n_hotspots: Number of hotspot hosts
            hotspot_rate: Rate for flows to/from hotspots
            background_rate: Rate for other flows
            background_sparsity: Sparsity for background traffic
        """
        tm = np.zeros((self.num_hosts, self.num_hosts))
        
        # Select hotspot hosts from different pods
        hotspots = []
        pods_used = set()
        for host in self.rng.permutation(self.topo.host_ids):
            pod = self.topo.pod_of_host[host]
            if pod not in pods_used:
                hotspots.append(host)
                pods_used.add(pod)
                if len(hotspots) >= n_hotspots:
                    break
        
        # Traffic to/from hotspots
        for hotspot in hotspots:
            for other in self.topo.host_ids:
                if other != hotspot:
                    hi = hotspot - self.topo.host_offset
                    oi = other - self.topo.host_offset
                    # Hotspot sends and receives
                    tm[hi, oi] = hotspot_rate
                    tm[oi, hi] = hotspot_rate
        
        # Background traffic
        for src in self.topo.host_ids:
            for dst in self.topo.host_ids:
                si, di = src - self.topo.host_offset, dst - self.topo.host_offset
                if src != dst and tm[si, di] == 0:
                    if self.rng.random() < background_sparsity:
                        tm[si, di] = background_rate
                        
        return tm
    
    def tm_to_flow_vector(self, tm: np.ndarray) -> np.ndarray:
        """Convert traffic matrix to flow vector (ordered by flow_pairs)."""
        flows = np.zeros(len(self.topo.flow_pairs))
        for flow_idx, (src, dst) in enumerate(self.topo.flow_pairs):
            flows[flow_idx] = tm[src - self.topo.host_offset, dst - self.topo.host_offset]
        return flows
    
    def flow_vector_to_tm(self, flows: np.ndarray) -> np.ndarray:
        """Convert flow vector back to traffic matrix."""
        tm = np.zeros((self.num_hosts, self.num_hosts))
        for flow_idx, (src, dst) in enumerate(self.topo.flow_pairs):
            tm[src - self.topo.host_offset, dst - self.topo.host_offset] = flows[flow_idx]
        return tm


def test_traffic_gen():
    """Test traffic generation."""
    print("="*60)
    print("FAT-TREE TRAFFIC GENERATOR TEST")
    print("="*60)
    
    topo = FatTreeTopology(k=4)
    gen = FatTreeTrafficGenerator(topo, seed=42)
    
    print(f"\nFlow categories:")
    print(f"  Same-edge flows: {len(gen.same_edge_flows)}")
    print(f"  Same-pod flows: {len(gen.same_pod_flows)}")
    print(f"  Cross-pod flows: {len(gen.cross_pod_flows)}")
    print(f"  Multi-path flows: {len(gen.multi_path_flows)}")
    
    # Test bimodal
    print("\n" + "-"*60)
    print("BIMODAL TRAFFIC")
    print("-"*60)
    tm = gen.generate_bimodal(n_elephants=8, elephant_rate=400e6, mouse_rate=50e6)
    flows = gen.tm_to_flow_vector(tm)
    
    n_active = np.sum(flows > 0)
    n_elephants = np.sum(flows >= 400e6)
    total = tm.sum()
    
    print(f"  Active flows: {n_active}")
    print(f"  Elephant flows: {n_elephants}")
    print(f"  Total traffic: {total/1e9:.2f} Gbps")
    
    # Count elephants by type
    elephant_cross_pod = 0
    for flow_idx in range(len(flows)):
        if flows[flow_idx] >= 400e6:
            src, dst = topo.flow_pairs[flow_idx]
            if not topo.hosts_same_pod(src, dst):
                elephant_cross_pod += 1
    print(f"  Cross-pod elephants: {elephant_cross_pod}/{n_elephants}")
    
    # Test hotspot
    print("\n" + "-"*60)
    print("HOTSPOT TRAFFIC")
    print("-"*60)
    tm_hot = gen.generate_hotspot(n_hotspots=2, hotspot_rate=300e6)
    print(f"  Total traffic: {tm_hot.sum()/1e9:.2f} Gbps")
    
    # Test uniform
    print("\n" + "-"*60)
    print("UNIFORM TRAFFIC")
    print("-"*60)
    tm_uni = gen.generate_uniform(rate=100e6, sparsity=0.3)
    print(f"  Total traffic: {tm_uni.sum()/1e9:.2f} Gbps")
    print(f"  Active flows: {np.sum(tm_uni > 0)}")


if __name__ == "__main__":
    test_traffic_gen()