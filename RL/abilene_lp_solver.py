"""
LP Solver for Abilene Topology

Same approach as fat-tree, but adapted for Abilene's:
- Asymmetric link capacities
- Variable path counts
- Different bottleneck patterns
"""

import numpy as np
from typing import List, Dict, Tuple
from ortools.linear_solver import pywraplp
from abilene_topology import AbileneToplogy


class AbileneLPSolver:
    """LP solver for Abilene routing optimization."""
    
    def __init__(self, topology: AbileneToplogy):
        self.topo = topology
        
    def solve(self,
              traffic_matrix: np.ndarray,
              critical_flows: List[int],
              background_routing: str = "single_path") -> Tuple[float, Dict]:
        """
        Solve routing LP with critical flows getting optimal multi-path routing.
        
        Args:
            traffic_matrix: Node x Node demand matrix (bps)
            critical_flows: Flow indices to optimize
            background_routing: "single_path" or "ecmp" for non-critical flows
            
        Returns:
            (max_utilization, routing_dict)
        """
        solver = pywraplp.Solver.CreateSolver('GLOP')
        if not solver:
            raise RuntimeError("Could not create GLOP solver")
        
        # Max utilization variable
        U = solver.NumVar(0, solver.infinity(), 'U')
        
        # Routing variables for critical flows only
        r = {}
        critical_set = set(critical_flows)
        
        for flow_idx in critical_flows:
            src, dst = self.topo.flow_pairs[flow_idx]
            num_paths = self.topo.get_num_paths(src, dst)
            
            if num_paths > 0:
                r[flow_idx] = [
                    solver.NumVar(0, 1, f'r_{flow_idx}_{p}')
                    for p in range(num_paths)
                ]
                # Must sum to 1
                solver.Add(sum(r[flow_idx]) == 1)
        
        # Compute link loads
        link_load = {link: 0.0 for link in self.topo.links}
        
        for flow_idx, (src, dst) in enumerate(self.topo.flow_pairs):
            demand = traffic_matrix[src, dst]
            if demand <= 0:
                continue
            
            paths = self.topo.get_all_paths(src, dst)
            num_paths = len(paths)
            
            if num_paths == 0:
                continue
            
            if flow_idx in critical_set and flow_idx in r:
                # Critical flow: LP-optimized routing
                for p_idx, path in enumerate(paths):
                    for link in self.topo.get_path_links(path):
                        link_load[link] = link_load[link] + demand * r[flow_idx][p_idx]
            else:
                # Background flow: fixed routing
                if background_routing == "single_path":
                    for link in self.topo.get_path_links(paths[0]):
                        link_load[link] = link_load[link] + demand
                else:  # ecmp
                    split = demand / num_paths
                    for path in paths:
                        for link in self.topo.get_path_links(path):
                            link_load[link] = link_load[link] + split
        
        # Capacity constraints
        for link in self.topo.links:
            cap = self.topo.get_link_capacity(*link)
            if cap > 0:
                solver.Add(link_load[link] <= cap * U)
        
        # Minimize max utilization
        solver.Minimize(U)
        
        status = solver.Solve()
        
        if status != pywraplp.Solver.OPTIMAL:
            return float('inf'), {}
        
        # Extract routing decisions
        routing = {}
        for flow_idx in critical_flows:
            if flow_idx in r:
                src, dst = self.topo.flow_pairs[flow_idx]
                num_paths = self.topo.get_num_paths(src, dst)
                routing[flow_idx] = [r[flow_idx][p].solution_value() for p in range(num_paths)]
        
        return U.solution_value(), routing
    
    def solve_single_path(self, traffic_matrix: np.ndarray) -> float:
        """All flows use shortest path (first path)."""
        link_load = {link: 0.0 for link in self.topo.links}
        
        for src, dst in self.topo.flow_pairs:
            demand = traffic_matrix[src, dst]
            if demand <= 0:
                continue
            
            paths = self.topo.get_all_paths(src, dst)
            if paths:
                for link in self.topo.get_path_links(paths[0]):
                    link_load[link] += demand
        
        max_util = 0.0
        for link, load in link_load.items():
            cap = self.topo.get_link_capacity(*link)
            if cap > 0:
                max_util = max(max_util, load / cap)
        
        return max_util
    
    def solve_ecmp(self, traffic_matrix: np.ndarray) -> float:
        """All flows use ECMP (equal split across all paths)."""
        link_load = {link: 0.0 for link in self.topo.links}
        
        for src, dst in self.topo.flow_pairs:
            demand = traffic_matrix[src, dst]
            if demand <= 0:
                continue
            
            paths = self.topo.get_all_paths(src, dst)
            if paths:
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
        """
        flow_demands = []
        for flow_idx, (src, dst) in enumerate(self.topo.flow_pairs):
            demand = traffic_matrix[src, dst]
            num_paths = self.topo.get_num_paths(src, dst)
            
            if multi_path_only and num_paths <= 1:
                continue
            
            flow_demands.append((flow_idx, demand))
        
        flow_demands.sort(key=lambda x: -x[1])
        top_k_flows = [f[0] for f in flow_demands[:k]]
        
        util, _ = self.solve(traffic_matrix, top_k_flows, background_routing="single_path")
        return util, top_k_flows
    
    def solve_random_k(self, traffic_matrix: np.ndarray, k: int,
                       multi_path_only: bool = True) -> Tuple[float, List[int]]:
        """Select random K flows and give them optimal routing."""
        eligible = []
        for flow_idx, (src, dst) in enumerate(self.topo.flow_pairs):
            demand = traffic_matrix[src, dst]
            if demand <= 0:
                continue
            num_paths = self.topo.get_num_paths(src, dst)
            if multi_path_only and num_paths <= 1:
                continue
            eligible.append(flow_idx)
        
        if len(eligible) <= k:
            selected = eligible
        else:
            selected = list(np.random.choice(eligible, k, replace=False))
        
        util, _ = self.solve(traffic_matrix, selected, background_routing="single_path")
        return util, selected
    
    def get_link_utilizations(self, traffic_matrix: np.ndarray,
                              critical_flows: List[int],
                              routing: Dict[int, List[float]]) -> Dict[Tuple[int, int], float]:
        """
        Compute per-link utilization given routing decisions.
        Useful for analysis.
        """
        link_load = {link: 0.0 for link in self.topo.links}
        critical_set = set(critical_flows)
        
        for flow_idx, (src, dst) in enumerate(self.topo.flow_pairs):
            demand = traffic_matrix[src, dst]
            if demand <= 0:
                continue
            
            paths = self.topo.get_all_paths(src, dst)
            if not paths:
                continue
            
            if flow_idx in critical_set and flow_idx in routing:
                ratios = routing[flow_idx]
                for p_idx, path in enumerate(paths):
                    for link in self.topo.get_path_links(path):
                        link_load[link] += demand * ratios[p_idx]
            else:
                for link in self.topo.get_path_links(paths[0]):
                    link_load[link] += demand
        
        utilizations = {}
        for link, load in link_load.items():
            cap = self.topo.get_link_capacity(*link)
            utilizations[link] = load / cap if cap > 0 else 0.0
        
        return utilizations


def test_abilene_lp():
    """Test LP solver on Abilene."""
    print("="*60)
    print("ABILENE LP SOLVER TEST")
    print("="*60)
    
    topo = AbileneToplogy()
    solver = AbileneLPSolver(topo)
    
    # Create test traffic
    np.random.seed(42)
    tm = np.zeros((topo.num_nodes, topo.num_nodes))
    
    # Generate traffic pattern
    # Elephants between distant nodes (cross-country)
    elephants = [
        (0, 10, 2e9),   # Seattle -> New York: 2 Gbps
        (2, 9, 1.5e9),  # LA -> Washington: 1.5 Gbps
        (1, 8, 1e9),    # Sunnyvale -> Atlanta: 1 Gbps
        (3, 10, 1e9),   # Denver -> New York: 1 Gbps
        (0, 8, 0.8e9),  # Seattle -> Atlanta: 800 Mbps
    ]
    
    for src, dst, rate in elephants:
        tm[src, dst] = rate
    
    # Add mice (smaller flows)
    for src in range(topo.num_nodes):
        for dst in range(topo.num_nodes):
            if src != dst and tm[src, dst] == 0:
                if np.random.random() < 0.3:
                    tm[src, dst] = np.random.uniform(50e6, 200e6)
    
    total_traffic = tm.sum()
    print(f"\nTraffic matrix:")
    print(f"  Total traffic: {total_traffic/1e9:.2f} Gbps")
    print(f"  Active flows: {np.sum(tm > 0)}")
    
    # Test routing strategies
    print("\n" + "-"*60)
    print("ROUTING COMPARISON")
    print("-"*60)
    
    util_single = solver.solve_single_path(tm)
    print(f"Single-Path:    {util_single:.4f}")
    
    util_ecmp = solver.solve_ecmp(tm)
    print(f"ECMP:           {util_ecmp:.4f}")
    
    util_random, _ = solver.solve_random_k(tm, k=8)
    print(f"Random-K (K=8): {util_random:.4f}")
    
    util_topk, topk_flows = solver.solve_top_k(tm, k=8)
    print(f"Top-K (K=8):    {util_topk:.4f}")
    
    # Analysis
    print("\n" + "-"*60)
    print("ANALYSIS")
    print("-"*60)
    
    if util_single > util_ecmp:
        improvement = (util_single - util_ecmp) / util_single * 100
        print(f"ECMP improvement over single-path: {improvement:.1f}%")
        
        topk_improvement = (util_single - util_topk) / util_single * 100
        print(f"Top-K improvement over single-path: {topk_improvement:.1f}%")
        
        if util_single > util_ecmp:
            gap_captured = (util_single - util_topk) / (util_single - util_ecmp) * 100
            print(f"Top-K captures {gap_captured:.1f}% of ECMP improvement")
    
    # Show Top-K selections
    print(f"\nTop-K selected flows:")
    for i, flow_idx in enumerate(topk_flows[:5]):
        src, dst = topo.flow_pairs[flow_idx]
        demand = tm[src, dst]
        num_paths = topo.get_num_paths(src, dst)
        print(f"  {i+1}. {topo.node_abbrev[src]} -> {topo.node_abbrev[dst]}: {demand/1e6:.0f} Mbps ({num_paths} paths)")


if __name__ == "__main__":
    test_abilene_lp()