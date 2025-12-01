"""
LP Solver for Fat-Tree Topology

Solves multi-commodity flow problem to minimize max link utilization.
Supports up to 4 paths per flow (cross-pod flows).
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from ortools.linear_solver import pywraplp
from fat_tree_topology import FatTreeTopology


class FatTreeLPSolver:
    """LP solver for routing optimization on fat-tree topology."""
    
    def __init__(self, topology: FatTreeTopology):
        self.topo = topology
        
    def solve(self, 
              traffic_matrix: np.ndarray,
              critical_flows: List[int],
              background_routing: str = "single_path") -> Tuple[float, Dict]:
        """
        Solve routing LP with critical flows getting optimal multi-path routing.
        
        Args:
            traffic_matrix: Host x Host demand matrix (in bps)
            critical_flows: List of flow indices to optimize (get multi-path)
            background_routing: How to route non-critical flows
                - "single_path": All via first path (worst case baseline)
                - "ecmp": Split equally across all paths
                
        Returns:
            (max_utilization, routing_dict)
        """
        solver = pywraplp.Solver.CreateSolver('GLOP')
        if not solver:
            raise RuntimeError("Could not create GLOP solver")
            
        # Decision variables
        U = solver.NumVar(0, solver.infinity(), 'U')  # Max utilization
        
        # Routing ratios for critical flows: r[flow_idx][path_idx]
        r = {}
        critical_set = set(critical_flows)
        
        for flow_idx in critical_flows:
            src, dst = self.topo.flow_pairs[flow_idx]
            num_paths = self.topo.get_num_paths(src, dst)
            r[flow_idx] = [
                solver.NumVar(0, 1, f'r_{flow_idx}_{p}')
                for p in range(num_paths)
            ]
            # Routing ratios must sum to 1
            solver.Add(sum(r[flow_idx]) == 1)
        
        # Compute link loads
        link_load = {link: 0.0 for link in self.topo.links}
        
        for flow_idx, (src, dst) in enumerate(self.topo.flow_pairs):
            demand = traffic_matrix[src - self.topo.host_offset, dst - self.topo.host_offset]
            if demand <= 0:
                continue
                
            paths = self.topo.get_all_paths(src, dst)
            num_paths = len(paths)
            
            if flow_idx in critical_set:
                # Critical flow: use LP variables
                for p_idx, path in enumerate(paths):
                    for link in self.topo.get_path_links(path):
                        link_load[link] = link_load[link] + demand * r[flow_idx][p_idx]
            else:
                # Background flow: fixed routing
                if background_routing == "single_path":
                    # Use first path only
                    for link in self.topo.get_path_links(paths[0]):
                        link_load[link] = link_load[link] + demand
                else:  # ecmp
                    # Split equally across all paths
                    split = demand / num_paths
                    for path in paths:
                        for link in self.topo.get_path_links(path):
                            link_load[link] = link_load[link] + split
        
        # Link capacity constraints
        for link in self.topo.links:
            cap = self.topo.get_link_capacity(*link)
            if cap > 0:
                solver.Add(link_load[link] <= cap * U)
        
        # Objective: minimize max utilization
        solver.Minimize(U)
        
        # Solve
        status = solver.Solve()
        
        if status != pywraplp.Solver.OPTIMAL:
            return float('inf'), {}
            
        # Extract routing
        routing = {}
        for flow_idx in critical_flows:
            src, dst = self.topo.flow_pairs[flow_idx]
            num_paths = self.topo.get_num_paths(src, dst)
            routing[flow_idx] = [r[flow_idx][p].solution_value() for p in range(num_paths)]
            
        return U.solution_value(), routing
    
    def solve_single_path(self, traffic_matrix: np.ndarray) -> float:
        """All flows use single path (first path) - worst case baseline."""
        link_load = {link: 0.0 for link in self.topo.links}
        
        for src, dst in self.topo.flow_pairs:
            demand = traffic_matrix[src - self.topo.host_offset, dst - self.topo.host_offset]
            if demand <= 0:
                continue
            paths = self.topo.get_all_paths(src, dst)
            for link in self.topo.get_path_links(paths[0]):
                link_load[link] += demand
                
        max_util = 0.0
        for link, load in link_load.items():
            cap = self.topo.get_link_capacity(*link)
            if cap > 0:
                max_util = max(max_util, load / cap)
                
        return max_util
    
    def solve_ecmp(self, traffic_matrix: np.ndarray) -> float:
        """All flows use ECMP (equal split) - best case."""
        link_load = {link: 0.0 for link in self.topo.links}
        
        for src, dst in self.topo.flow_pairs:
            demand = traffic_matrix[src - self.topo.host_offset, dst - self.topo.host_offset]
            if demand <= 0:
                continue
            paths = self.topo.get_all_paths(src, dst)
            split = demand / len(paths)
            for path in paths:
                for link in self.topo.get_path_links(path):
                    link_load[link] += split
                    
        max_util = 0.0
        for link, load in link_load.items():
            cap = self.topo.get_link_capacity(*link)
            if cap > 0:
                max_util = max(max_util, load / cap)
                
        return max_util
    
    def solve_top_k(self, traffic_matrix: np.ndarray, k: int, 
                    multi_path_only: bool = True) -> Tuple[float, List[int]]:
        """
        Select top-K flows by demand and give them optimal routing.
        
        Args:
            traffic_matrix: Demand matrix
            k: Number of flows to optimize
            multi_path_only: If True, only consider flows with >1 path
            
        Returns:
            (max_utilization, selected_flow_indices)
        """
        # Get flow demands
        flow_demands = []
        for flow_idx, (src, dst) in enumerate(self.topo.flow_pairs):
            demand = traffic_matrix[src - self.topo.host_offset, dst - self.topo.host_offset]
            num_paths = self.topo.get_num_paths(src, dst)
            
            if multi_path_only and num_paths <= 1:
                continue
            flow_demands.append((flow_idx, demand))
        
        # Sort by demand and take top K
        flow_demands.sort(key=lambda x: -x[1])
        top_k_flows = [f[0] for f in flow_demands[:k]]
        
        # Solve with these as critical flows
        util, _ = self.solve(traffic_matrix, top_k_flows, background_routing="single_path")
        
        return util, top_k_flows
    
    def solve_random_k(self, traffic_matrix: np.ndarray, k: int,
                       multi_path_only: bool = True) -> Tuple[float, List[int]]:
        """
        Select random K flows and give them optimal routing.
        """
        # Get eligible flows
        eligible = []
        for flow_idx, (src, dst) in enumerate(self.topo.flow_pairs):
            demand = traffic_matrix[src - self.topo.host_offset, dst - self.topo.host_offset]
            if demand <= 0:
                continue
            num_paths = self.topo.get_num_paths(src, dst)
            if multi_path_only and num_paths <= 1:
                continue
            eligible.append(flow_idx)
        
        # Random selection
        if len(eligible) <= k:
            selected = eligible
        else:
            selected = list(np.random.choice(eligible, k, replace=False))
        
        # Solve
        util, _ = self.solve(traffic_matrix, selected, background_routing="single_path")
        
        return util, selected


def test_fat_tree_lp():
    """Test LP solver on fat-tree topology."""
    print("="*60)
    print("FAT-TREE LP SOLVER TEST")
    print("="*60)
    
    topo = FatTreeTopology(k=4)
    solver = FatTreeLPSolver(topo)
    
    # Create test traffic matrix
    # Focus traffic on cross-pod flows to stress the core
    np.random.seed(42)
    num_hosts = topo.num_hosts
    tm = np.zeros((num_hosts, num_hosts))
    
    # Generate bimodal traffic (elephants + mice)
    n_elephants = 8
    elephant_rate = 400e6  # 400 Mbps
    mouse_rate = 50e6      # 50 Mbps
    
    # Elephants: random cross-pod pairs
    cross_pod_pairs = [
        (src, dst) for src in topo.host_ids for dst in topo.host_ids
        if src != dst and not topo.hosts_same_pod(src, dst)
    ]
    elephant_pairs = np.random.choice(len(cross_pod_pairs), n_elephants, replace=False)
    
    for idx in elephant_pairs:
        src, dst = cross_pod_pairs[idx]
        tm[src - topo.host_offset, dst - topo.host_offset] = elephant_rate
    
    # Mice: sparse random traffic
    for src in topo.host_ids:
        for dst in topo.host_ids:
            if src != dst and tm[src - topo.host_offset, dst - topo.host_offset] == 0:
                if np.random.random() < 0.3:  # 30% chance
                    tm[src - topo.host_offset, dst - topo.host_offset] = mouse_rate
    
    total_traffic = tm.sum()
    print(f"\nTraffic matrix:")
    print(f"  Total traffic: {total_traffic/1e9:.2f} Gbps")
    print(f"  Elephants: {n_elephants} @ {elephant_rate/1e6:.0f} Mbps")
    print(f"  Core-agg bandwidth: {topo.link_capacity[(0, 4)]/1e9:.1f} Gbps per link")
    print(f"  Total core capacity: {4 * topo.link_capacity[(0, 4)]/1e9:.1f} Gbps")
    
    # Test different routing strategies
    print("\n" + "-"*60)
    print("ROUTING COMPARISON")
    print("-"*60)
    
    # Single path (baseline)
    util_single = solver.solve_single_path(tm)
    print(f"Single-Path:     {util_single:.4f}")
    
    # ECMP (upper bound)
    util_ecmp = solver.solve_ecmp(tm)
    print(f"ECMP:            {util_ecmp:.4f}")
    
    # Random-K
    util_random, _ = solver.solve_random_k(tm, k=8)
    print(f"Random-K (K=8):  {util_random:.4f}")
    
    # Top-K
    util_topk, topk_flows = solver.solve_top_k(tm, k=8)
    print(f"Top-K (K=8):     {util_topk:.4f}")
    
    # CFR-RL would select flows via policy - for now just use top-K
    util_cfr, _ = solver.solve(tm, topk_flows, background_routing="single_path")
    print(f"CFR-RL (K=8):    {util_cfr:.4f}")
    
    # Analysis
    print("\n" + "-"*60)
    print("ANALYSIS")
    print("-"*60)
    improvement_ecmp = (util_single - util_ecmp) / util_single * 100
    improvement_topk = (util_single - util_topk) / util_single * 100
    print(f"ECMP improvement over single-path:  {improvement_ecmp:.1f}%")
    print(f"Top-K improvement over single-path: {improvement_topk:.1f}%")
    print(f"Top-K captures {improvement_topk/improvement_ecmp*100:.1f}% of ECMP improvement")
    
    # Show which flows Top-K selected
    print(f"\nTop-K selected flows:")
    for i, flow_idx in enumerate(topk_flows[:5]):
        src, dst = topo.flow_pairs[flow_idx]
        demand = tm[src - topo.host_offset, dst - topo.host_offset]
        ftype = topo.get_flow_type(src, dst)
        num_paths = topo.get_num_paths(src, dst)
        print(f"  {i+1}. Flow {flow_idx}: {src}->{dst} ({ftype}, {num_paths} paths) = {demand/1e6:.0f} Mbps")


if __name__ == "__main__":
    test_fat_tree_lp()