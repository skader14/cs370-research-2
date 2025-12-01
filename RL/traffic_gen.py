"""
Traffic matrix generator following Zhang et al. CFR-RL paper.
Generates synthetic traffic demands with bias toward cross-edge flows.
"""
import numpy as np
from typing import List, Tuple, Optional
from topology import Topology


class TrafficGenerator:
    """
    Generate traffic matrices for training.
    
    Traffic matrix T where T[i,j] = demand from host i to host j (in bps).
    Diagonal is always 0 (no self-traffic).
    """
    
    def __init__(self, topology: Topology, seed: Optional[int] = None):
        self.topo = topology
        self.n_hosts = topology.n_hosts
        self.host_ids = topology.host_ids
        self.rng = np.random.default_rng(seed)
        
        # Precompute which flows are cross-edge (have multiple paths)
        self.cross_edge_flows = []
        self.same_edge_flows = []
        for i in range(self.n_hosts):
            for j in range(self.n_hosts):
                if i != j:
                    host_i = self.host_ids[i]
                    host_j = self.host_ids[j]
                    if self.topo.hosts_same_edge(host_i, host_j):
                        self.same_edge_flows.append((i, j))
                    else:
                        self.cross_edge_flows.append((i, j))
        
    def generate_cross_edge_biased(self, mean_demand: float = 100e6,
                                   cross_edge_prob: float = 0.8,
                                   sparsity: float = 0.4) -> np.ndarray:
        """
        Generate traffic biased toward cross-edge flows (which can be optimized).
        
        Args:
            mean_demand: Mean demand in bps
            cross_edge_prob: Probability of activating cross-edge vs same-edge flows
            sparsity: Overall fraction of flows that are active
        """
        tm = np.zeros((self.n_hosts, self.n_hosts))
        
        # Activate cross-edge flows with higher probability
        for (i, j) in self.cross_edge_flows:
            if self.rng.random() < sparsity * cross_edge_prob:
                tm[i, j] = self.rng.exponential(mean_demand)
        
        # Activate same-edge flows with lower probability
        for (i, j) in self.same_edge_flows:
            if self.rng.random() < sparsity * (1 - cross_edge_prob) * 0.5:
                tm[i, j] = self.rng.exponential(mean_demand * 0.5)  # Smaller demands
        
        return tm
    
    def generate_bimodal(self, n_elephant: int = 5, 
                    elephant_demand: float = 600e6,  # Increased
                    mice_demand: float = 50e6,       # Increased
                    sparsity: float = 0.5,           # More flows
                    cross_edge_only_elephants: bool = True) -> np.ndarray:
        """
        Generate bimodal traffic: few elephant flows + many mice flows.
        Traffic is scaled to cause congestion on core links.
        """
        tm = np.zeros((self.n_hosts, self.n_hosts))
        
        # Mice flows - cross-edge
        for (i, j) in self.cross_edge_flows:
            if self.rng.random() < sparsity:
                tm[i, j] = self.rng.exponential(mice_demand)
        
        # Small amount of same-edge mice
        for (i, j) in self.same_edge_flows:
            if self.rng.random() < sparsity * 0.1:
                tm[i, j] = self.rng.exponential(mice_demand * 0.3)
        
        # Elephant flows - cross-edge only (these will cause congestion!)
        if cross_edge_only_elephants:
            elephant_candidates = self.cross_edge_flows
        else:
            elephant_candidates = self.cross_edge_flows + self.same_edge_flows
            
        n_elephant = min(n_elephant, len(elephant_candidates))
        elephant_indices = self.rng.choice(len(elephant_candidates), 
                                        size=n_elephant, replace=False)
        
        for idx in elephant_indices:
            i, j = elephant_candidates[idx]
            # Large elephant flows that will stress the network
            tm[i, j] = elephant_demand * self.rng.uniform(0.9, 1.1)
                    
        return tm
    
    def generate_exponential(self, mean_demand: float = 100e6, 
                            sparsity: float = 0.3) -> np.ndarray:
        """Generate traffic with exponential distribution, biased cross-edge."""
        return self.generate_cross_edge_biased(mean_demand, 
                                               cross_edge_prob=0.8, 
                                               sparsity=sparsity)
    
    def generate_gravity(self, total_traffic: float = 1e9,
                        sparsity: float = 0.3) -> np.ndarray:
        """Generate using gravity model, biased cross-edge."""
        # Random weights for each host
        weights = self.rng.exponential(1.0, self.n_hosts)
        weights = weights / weights.sum()
        
        tm = np.zeros((self.n_hosts, self.n_hosts))
        
        # Cross-edge flows get full gravity weight
        for (i, j) in self.cross_edge_flows:
            if self.rng.random() < sparsity:
                tm[i, j] = weights[i] * weights[j] * total_traffic * 10
        
        # Same-edge flows get reduced weight
        for (i, j) in self.same_edge_flows:
            if self.rng.random() < sparsity * 0.3:
                tm[i, j] = weights[i] * weights[j] * total_traffic * 2
        
        return tm
    
    def generate_batch(self, n_matrices: int, method: str = "bimodal",
                      **kwargs) -> List[np.ndarray]:
        """Generate multiple traffic matrices."""
        gen_func = {
            "exponential": self.generate_exponential,
            "gravity": self.generate_gravity,
            "bimodal": self.generate_bimodal,
        }[method]
        
        return [gen_func(**kwargs) for _ in range(n_matrices)]
    
    def tm_to_flow_vector(self, tm: np.ndarray) -> np.ndarray:
        """Flatten traffic matrix to flow vector (excluding diagonal)."""
        n = self.n_hosts
        flows = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    flows.append(tm[i, j])
        return np.array(flows)
    
    def flow_vector_to_tm(self, flows: np.ndarray) -> np.ndarray:
        """Convert flow vector back to traffic matrix."""
        n = self.n_hosts
        tm = np.zeros((n, n))
        idx = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    tm[i, j] = flows[idx]
                    idx += 1
        return tm
    
    def normalize_flows(self, tm: np.ndarray) -> np.ndarray:
        """Normalize flow vector by total traffic."""
        flows = self.tm_to_flow_vector(tm)
        total = flows.sum()
        if total > 0:
            flows = flows / total
        return flows


def test_traffic_gen():
    """Test traffic generation."""
    topo = Topology(n_edge=4, hosts_per_edge=4)
    gen = TrafficGenerator(topo, seed=42)
    
    print(f"Cross-edge flows: {len(gen.cross_edge_flows)}")
    print(f"Same-edge flows: {len(gen.same_edge_flows)}")
    
    # Generate traffic
    tm = gen.generate_bimodal(n_elephant=5, elephant_demand=400e6)
    print(f"\nTraffic matrix shape: {tm.shape}")
    print(f"Total traffic: {tm.sum() / 1e9:.2f} Gbps")
    print(f"Active flows: {np.count_nonzero(tm)}")
    
    # Count cross-edge vs same-edge active traffic
    cross_edge_traffic = sum(tm[i, j] for (i, j) in gen.cross_edge_flows)
    same_edge_traffic = sum(tm[i, j] for (i, j) in gen.same_edge_flows)
    print(f"Cross-edge traffic: {cross_edge_traffic / 1e9:.2f} Gbps ({cross_edge_traffic / tm.sum() * 100:.1f}%)")
    print(f"Same-edge traffic: {same_edge_traffic / 1e9:.2f} Gbps ({same_edge_traffic / tm.sum() * 100:.1f}%)")
    
    # Show largest flows
    flows = []
    for i in range(gen.n_hosts):
        for j in range(gen.n_hosts):
            if tm[i, j] > 0:
                is_cross = (i, j) in gen.cross_edge_flows
                flows.append((i, j, tm[i, j], is_cross))
    
    flows.sort(key=lambda x: -x[2])
    print(f"\nTop 5 flows:")
    for src, dst, demand, is_cross in flows[:5]:
        path_type = "cross-edge (2 paths)" if is_cross else "same-edge (1 path)"
        print(f"  ({src}, {dst}): {demand/1e6:.1f} Mbps - {path_type}")


if __name__ == "__main__":
    test_traffic_gen()