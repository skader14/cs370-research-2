"""
LP Solver for Critical Flow Routing with multi-path support.
Compares: Single-path (baseline) vs ECMP vs Optimal LP routing.
"""
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from ortools.linear_solver import pywraplp
from topology import Topology


class LPSolver:
    """
    Solves the traffic engineering LP for critical flow routing.
    
    Baseline comparison:
    - Single-path: All traffic via core 0 only (like OSPF shortest path)
    - ECMP: 50/50 split across both cores
    - Optimal: LP-optimized routing for critical flows
    
    The value of CFR-RL is selecting which flows to give multi-path routing.
    """
    
    def __init__(self, topology: Topology):
        self.topo = topology
        
    def compute_single_path_load(self, tm: np.ndarray,
                                  exclude_flows: Optional[Set[Tuple[int, int]]] = None
                                  ) -> Dict[Tuple[int, int], float]:
        """
        Compute link loads using SINGLE PATH routing (all via core 0).
        This is the baseline that CFR-RL improves upon.
        """
        exclude_flows = exclude_flows or set()
        link_load: Dict[Tuple[int, int], float] = {link: 0.0 for link in self.topo.links}
        
        n_hosts = self.topo.n_hosts
        host_ids = self.topo.host_ids
        
        for i in range(n_hosts):
            for j in range(n_hosts):
                if i == j or (i, j) in exclude_flows:
                    continue
                    
                demand = tm[i, j]
                if demand <= 0:
                    continue
                    
                src_host = host_ids[i]
                dst_host = host_ids[j]
                paths = self.topo.get_all_paths(src_host, dst_host)
                
                # Single path: use first path only (via core 0)
                path = paths[0]
                path_links = self.topo.get_path_links(path)
                for link in path_links:
                    link_load[link] += demand
                    
        return link_load
    
    def compute_ecmp_load(self, tm: np.ndarray, 
                         exclude_flows: Optional[Set[Tuple[int, int]]] = None
                         ) -> Dict[Tuple[int, int], float]:
        """
        Compute link loads using ECMP (equal split across all paths).
        """
        exclude_flows = exclude_flows or set()
        link_load: Dict[Tuple[int, int], float] = {link: 0.0 for link in self.topo.links}
        
        n_hosts = self.topo.n_hosts
        host_ids = self.topo.host_ids
        
        for i in range(n_hosts):
            for j in range(n_hosts):
                if i == j or (i, j) in exclude_flows:
                    continue
                    
                demand = tm[i, j]
                if demand <= 0:
                    continue
                    
                src_host = host_ids[i]
                dst_host = host_ids[j]
                paths = self.topo.get_all_paths(src_host, dst_host)
                
                # ECMP: equal split
                demand_per_path = demand / len(paths)
                for path in paths:
                    path_links = self.topo.get_path_links(path)
                    for link in path_links:
                        link_load[link] += demand_per_path
                    
        return link_load
    
    def get_max_utilization(self, link_load: Dict[Tuple[int, int], float]) -> float:
        """Compute max link utilization from link loads."""
        max_util = 0.0
        for link, load in link_load.items():
            cap = self.topo.link_capacity[link]
            util = load / cap if cap > 0 else 0
            max_util = max(max_util, util)
        return max_util
    
    def solve_single_path(self, tm: np.ndarray) -> Tuple[float, Dict]:
        """Baseline: all flows use single path (core 0 only)."""
        link_load = self.compute_single_path_load(tm)
        max_util = self.get_max_utilization(link_load)
        return max_util, {"link_loads": link_load, "method": "single_path"}
    
    def solve_ecmp(self, tm: np.ndarray) -> Tuple[float, Dict]:
        """ECMP: all flows split 50/50 across paths."""
        link_load = self.compute_ecmp_load(tm)
        max_util = self.get_max_utilization(link_load)
        return max_util, {"link_loads": link_load, "method": "ecmp"}
    
    def solve(self, tm: np.ndarray, critical_flows: List[Tuple[int, int]]
             ) -> Tuple[float, Dict]:
        """
        CFR-RL approach: 
        - Non-critical flows use SINGLE PATH (baseline routing)
        - Critical flows get LP-optimized multi-path routing
        
        This models the real scenario where:
        - Default routing is single-path (OSPF)
        - SDN controller can install multi-path rules for selected flows
        """
        # If no critical flows, use single-path baseline
        if not critical_flows:
            return self.solve_single_path(tm)
        
        # Background load: non-critical flows use SINGLE PATH
        exclude_set = set(critical_flows)
        background_load = self.compute_single_path_load(tm, exclude_set)
        
        # Create LP solver for critical flows
        solver = pywraplp.Solver.CreateSolver('GLOP')
        if not solver:
            raise RuntimeError("Could not create GLOP solver")
        
        # Decision variables
        U = solver.NumVar(0, solver.infinity(), 'U')
        
        host_ids = self.topo.host_ids
        flow_path_vars = {}
        flow_paths = {}
        
        for (src_idx, dst_idx) in critical_flows:
            src_host = host_ids[src_idx]
            dst_host = host_ids[dst_idx]
            paths = self.topo.get_all_paths(src_host, dst_host)
            flow_paths[(src_idx, dst_idx)] = paths
            
            path_vars = {}
            for p_idx in range(len(paths)):
                var_name = f'r_{src_idx}_{dst_idx}_{p_idx}'
                path_vars[p_idx] = solver.NumVar(0, 1, var_name)
            flow_path_vars[(src_idx, dst_idx)] = path_vars
            
            # Routing ratios sum to 1
            solver.Add(sum(path_vars.values()) == 1)
        
        # Link capacity constraints
        for link in self.topo.links:
            link_load_expr = background_load[link]
            
            for (src_idx, dst_idx) in critical_flows:
                demand = tm[src_idx, dst_idx]
                if demand <= 0:
                    continue
                    
                paths = flow_paths[(src_idx, dst_idx)]
                path_vars = flow_path_vars[(src_idx, dst_idx)]
                
                for p_idx, path in enumerate(paths):
                    path_links = self.topo.get_path_links(path)
                    if link in path_links:
                        link_load_expr += demand * path_vars[p_idx]
            
            cap = self.topo.link_capacity[link]
            solver.Add(link_load_expr <= cap * U)
        
        solver.Minimize(U)
        status = solver.Solve()
        
        if status != pywraplp.Solver.OPTIMAL:
            return 10.0, {"status": "infeasible"}
        
        max_util = U.solution_value()
        
        # Extract routing
        routing = {}
        for (src_idx, dst_idx) in critical_flows:
            paths = flow_paths[(src_idx, dst_idx)]
            path_vars = flow_path_vars[(src_idx, dst_idx)]
            ratios = [path_vars[p_idx].solution_value() for p_idx in range(len(paths))]
            routing[(src_idx, dst_idx)] = {"paths": paths, "ratios": ratios}
        
        # Compute actual link loads
        link_loads = dict(background_load)
        for (src_idx, dst_idx) in critical_flows:
            demand = tm[src_idx, dst_idx]
            if demand <= 0:
                continue
            paths = flow_paths[(src_idx, dst_idx)]
            ratios = routing[(src_idx, dst_idx)]["ratios"]
            for p_idx, path in enumerate(paths):
                for link in self.topo.get_path_links(path):
                    link_loads[link] += demand * ratios[p_idx]
        
        return max_util, {
            "routing": routing,
            "link_loads": link_loads,
            "status": "optimal",
        }
    
    def compute_reward(self, tm: np.ndarray, 
                      critical_flows: List[Tuple[int, int]]) -> float:
        """Reward = 1 / max_utilization."""
        max_util, _ = self.solve(tm, critical_flows)
        if max_util < 1e-6:
            return 100.0
        return 1.0 / max_util


def test_lp_solver():
    """Test LP solver showing single-path vs ECMP vs Optimal."""
    from traffic_gen import TrafficGenerator
    
    topo = Topology(n_edge=4, hosts_per_edge=4)
    gen = TrafficGenerator(topo, seed=42)
    solver = LPSolver(topo)
    
    print("=" * 60)
    print("TOPOLOGY")
    print("=" * 60)
    print(f"Core-edge link bandwidth: {topo.core_bw / 1e9:.1f} Gbps")
    print(f"Edge-host link bandwidth: {topo.edge_bw / 1e9:.1f} Gbps")
    print(f"Total core capacity: {topo.core_bw * topo.n_edge * 2 / 1e9:.1f} Gbps")
    
    # Generate traffic
    tm = gen.generate_bimodal(
        n_elephant=8,
        elephant_demand=600e6,
        mice_demand=60e6,
        sparsity=0.5
    )
    
    print(f"\n" + "=" * 60)
    print("TRAFFIC")
    print("=" * 60)
    print(f"Total traffic: {tm.sum() / 1e9:.2f} Gbps")
    
    # Get top flows
    flows_with_demand = []
    for i in range(topo.n_hosts):
        for j in range(topo.n_hosts):
            if i != j and tm[i, j] > 0:
                is_cross = (i, j) in gen.cross_edge_flows
                flows_with_demand.append((i, j, tm[i, j], is_cross))
    flows_with_demand.sort(key=lambda x: -x[2])
    
    print(f"\nTop 5 flows:")
    for src, dst, demand, is_cross in flows_with_demand[:5]:
        paths = 2 if is_cross else 1
        print(f"  ({src:2d}, {dst:2d}): {demand/1e6:6.1f} Mbps, {paths} path(s)")
    
    # === BASELINE: Single Path ===
    util_single, info_single = solver.solve_single_path(tm)
    
    # === ECMP: All flows ===
    util_ecmp, info_ecmp = solver.solve_ecmp(tm)
    
    # === CFR-RL: Critical flows get LP routing ===
    # Select top cross-edge flows
    cross_edge_sorted = [(f[0], f[1], f[2]) for f in flows_with_demand if f[3]]
    critical = [(f[0], f[1]) for f in cross_edge_sorted[:8]]
    critical_demand = sum(f[2] for f in cross_edge_sorted[:8])
    
    util_cfr, info_cfr = solver.solve(tm, critical)
    
    print(f"\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'Method':<25} {'Max Util':>10} {'Reward':>10}")
    print("-" * 60)
    print(f"{'Single-Path (baseline)':<25} {util_single:>10.4f} {1/util_single:>10.4f}")
    print(f"{'ECMP (all flows)':<25} {util_ecmp:>10.4f} {1/util_ecmp:>10.4f}")
    print(f"{'CFR-RL (8 critical)':<25} {util_cfr:>10.4f} {1/util_cfr:>10.4f}")
    
    # Improvements
    imp_ecmp = (util_single - util_ecmp) / util_single * 100
    imp_cfr = (util_single - util_cfr) / util_single * 100
    
    print(f"\n" + "=" * 60)
    print("IMPROVEMENT vs Single-Path Baseline")
    print("=" * 60)
    print(f"ECMP:   {imp_ecmp:+.1f}% utilization reduction")
    print(f"CFR-RL: {imp_cfr:+.1f}% utilization reduction")
    
    # Show routing for critical flows
    if "routing" in info_cfr:
        print(f"\n" + "=" * 60)
        print("CFR-RL ROUTING DECISIONS")
        print("=" * 60)
        for flow, data in list(info_cfr["routing"].items())[:5]:
            ratios = data["ratios"]
            if len(ratios) > 1:
                print(f"  Flow {flow}: Core0={ratios[0]*100:5.1f}%, Core1={ratios[1]*100:5.1f}%")


if __name__ == "__main__":
    test_lp_solver()