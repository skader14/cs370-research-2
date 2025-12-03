"""
LP Solver for Abilene Topology - LATENCY-AWARE VERSION

============================================================================
WHAT THIS FILE DOES
============================================================================

This is a modified version of the LP solver that supports a latency-aware
objective. The key change is in the solve() method's objective function.

ORIGINAL OBJECTIVE (Zhang et al. CFR-RL paper):
    minimize U
    
    where U = max link utilization across all links
    
    This gives optimal throughput but may concentrate traffic on few paths,
    causing high queuing delays on those paths.

NEW OBJECTIVE (latency-aware):
    minimize U + latency_weight * (average link utilization)
    
    The average utilization term encourages spreading load across links.
    When load is spread, no single link has high utilization, which means
    lower queuing delays (from queuing theory: delay ∝ util/(1-util)).

PARAMETER: latency_weight
    - 0.0 = Pure MLU optimization (original paper)
    - 0.1 = Slight latency awareness
    - 0.3 = Moderate latency awareness  
    - 0.5 = Strong latency awareness
    
    Higher values sacrifice some MLU for better latency.

============================================================================
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from ortools.linear_solver import pywraplp
from abilene_topology import AbileneToplogy
import logging
import json
from dataclasses import dataclass, asdict
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AbileneLPSolver")


@dataclass
class SolveResult:
    """Detailed result from LP solve - for logging and analysis."""
    max_utilization: float       # The MLU value (what original paper optimizes)
    total_utilization: float     # Sum of all link utilizations
    avg_utilization: float       # Average link utilization (proxy for latency)
    latency_term: float          # The latency term in the objective
    solve_time_ms: float         # How long the solve took
    solver_status: str           # OPTIMAL, INFEASIBLE, etc.
    latency_weight: float        # What weight was used
    num_critical_flows: int      # How many flows were optimized


class AbileneLPSolverLatencyAware:
    """
    LP solver for Abilene routing optimization with latency awareness.
    
    USAGE:
        solver = AbileneLPSolverLatencyAware(topology)
        
        # MLU-only (original paper)
        util, routing = solver.solve(tm, critical_flows, latency_weight=0.0)
        
        # Latency-aware
        util, routing = solver.solve(tm, critical_flows, latency_weight=0.3)
    """
    
    def __init__(self, topology: AbileneToplogy, verbose: bool = False):
        self.topo = topology
        self.verbose = verbose
        self.solve_count = 0
        self.solve_history: List[SolveResult] = []
        
        logger.info(f"LP Solver initialized: {topology.num_nodes} nodes, {topology.num_flows} flows")
    
    def solve(self,
              traffic_matrix: np.ndarray,
              critical_flows: List[int],
              background_routing: str = "single_path",
              latency_weight: float = 0.0) -> Tuple[float, Dict]:
        """
        Solve the routing LP with optional latency-aware objective.
        
        ========================================================================
        PARAMETERS
        ========================================================================
        
        traffic_matrix: np.ndarray
            NxN matrix where traffic_matrix[i,j] = demand from node i to node j
            Units: bytes per second (or any consistent unit)
        
        critical_flows: List[int]
            Indices of flows to give multi-path routing.
            These are the flows the RL agent selected for optimization.
            Other flows use fixed routing (background_routing).
        
        background_routing: str
            How to route non-critical flows:
            - "single_path": Use shortest path (first path)
            - "ecmp": Split equally across all paths
        
        latency_weight: float
            Weight for the latency term in the objective.
            - 0.0: Pure MLU minimization (original CFR-RL)
            - >0:  Trade some MLU for better latency
        
        ========================================================================
        RETURNS
        ========================================================================
        
        (max_utilization, routing_dict)
        
        max_utilization: float
            The MLU value achieved
        
        routing_dict: Dict[int, List[float]]
            For each critical flow, the split ratios across its paths.
            routing_dict[flow_idx] = [ratio_path0, ratio_path1, ...]
            Ratios sum to 1.0
        
        ========================================================================
        THE LP FORMULATION
        ========================================================================
        
        Variables:
            U: max link utilization (scalar)
            r[f][p]: fraction of flow f on path p (for critical flows only)
        
        Constraints:
            For each link L:
                sum of (demand * routing) for all flows using L <= capacity_L * U
            
            For each critical flow f:
                sum of r[f][p] for all paths p = 1 (all traffic must be routed)
        
        Objective:
            If latency_weight = 0:
                minimize U
            
            If latency_weight > 0:
                minimize U + latency_weight * (avg link utilization)
                
                where avg_utilization = (1/num_links) * sum(load_L / capacity_L)
        
        ========================================================================
        """
        import time
        start_time = time.time()
        self.solve_count += 1
        
        if self.verbose:
            logger.info(f"Solve #{self.solve_count}: {len(critical_flows)} critical flows, "
                       f"latency_weight={latency_weight}")
        
        # Create solver
        solver = pywraplp.Solver.CreateSolver('GLOP')
        if not solver:
            raise RuntimeError("Could not create GLOP solver")
        
        # =====================================================================
        # VARIABLE: U = max link utilization
        # =====================================================================
        U = solver.NumVar(0, solver.infinity(), 'U')
        
        # =====================================================================
        # VARIABLES: r[f][p] = routing ratios for critical flows
        # =====================================================================
        r = {}  # r[flow_idx] = list of variables, one per path
        critical_set = set(critical_flows)
        
        for flow_idx in critical_flows:
            src, dst = self.topo.flow_pairs[flow_idx]
            num_paths = self.topo.get_num_paths(src, dst)
            
            if num_paths > 0:
                # Create a variable for each path
                r[flow_idx] = [
                    solver.NumVar(0, 1, f'r_{flow_idx}_{p}')
                    for p in range(num_paths)
                ]
                # CONSTRAINT: ratios must sum to 1 (all traffic routed)
                solver.Add(sum(r[flow_idx]) == 1)
        
        # =====================================================================
        # COMPUTE LINK LOADS
        # This is an expression (sum of variables) for each link
        # =====================================================================
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
                # Critical flow: use LP variables for routing (multi-path)
                for p_idx, path in enumerate(paths):
                    for link in self.topo.get_path_links(path):
                        link_load[link] = link_load[link] + demand * r[flow_idx][p_idx]
            else:
                # Background flow: fixed routing
                if background_routing == "single_path":
                    # All traffic on first (shortest) path
                    for link in self.topo.get_path_links(paths[0]):
                        link_load[link] = link_load[link] + demand
                else:  # ecmp
                    # Split equally across all paths
                    split = demand / num_paths
                    for path in paths:
                        for link in self.topo.get_path_links(path):
                            link_load[link] = link_load[link] + split
        
        # =====================================================================
        # CONSTRAINTS: link load <= capacity * U (defines U as max utilization)
        # =====================================================================
        for link in self.topo.links:
            cap = self.topo.get_link_capacity(*link)
            if cap > 0:
                solver.Add(link_load[link] <= cap * U)
        
        # =====================================================================
        # OBJECTIVE FUNCTION
        # =====================================================================
        
        if latency_weight > 0:
            # -----------------------------------------------------------------
            # LATENCY-AWARE OBJECTIVE
            # -----------------------------------------------------------------
            # minimize U + latency_weight * (average link utilization)
            #
            # WHY THIS WORKS:
            # ---------------
            # From queuing theory (M/M/1 queue):
            #   delay = service_time / (1 - utilization)
            # 
            # When utilization is high, delay explodes. By minimizing the
            # average utilization, we encourage spreading load across links,
            # which keeps individual link utilizations lower.
            #
            # A more accurate model would use sum of util/(1-util), but that's
            # nonlinear. The sum of utilizations is a good linear proxy.
            # -----------------------------------------------------------------
            
            # Count links with positive capacity
            valid_links = [l for l in self.topo.links if self.topo.get_link_capacity(*l) > 0]
            num_links = len(valid_links)
            
            # Sum of utilizations
            utilization_sum = solver.Sum([
                link_load[link] / self.topo.get_link_capacity(*link)
                for link in valid_links
            ])
            
            # Average utilization (normalized)
            avg_utilization = utilization_sum / num_links
            
            # Combined objective: MLU + weighted latency term
            solver.Minimize(U + latency_weight * avg_utilization)
            
            if self.verbose:
                logger.debug(f"Objective: minimize(U + {latency_weight} * avg_util)")
        
        else:
            # -----------------------------------------------------------------
            # MLU-ONLY OBJECTIVE (original Zhang paper)
            # -----------------------------------------------------------------
            solver.Minimize(U)
            
            if self.verbose:
                logger.debug("Objective: minimize(U) [MLU only]")
        
        # =====================================================================
        # SOLVE
        # =====================================================================
        status = solver.Solve()
        solve_time_ms = (time.time() - start_time) * 1000
        
        if status != pywraplp.Solver.OPTIMAL:
            logger.warning(f"Solver status: {status} (not optimal)")
            return float('inf'), {}
        
        # =====================================================================
        # EXTRACT RESULTS
        # =====================================================================
        
        max_util = U.solution_value()
        
        # Extract routing decisions for critical flows
        routing = {}
        for flow_idx in critical_flows:
            if flow_idx in r:
                routing[flow_idx] = [r[flow_idx][p].solution_value() 
                                    for p in range(len(r[flow_idx]))]
        
        # Compute statistics for logging
        link_utils = self._compute_link_utilizations(
            traffic_matrix, critical_flows, routing, background_routing)
        util_values = list(link_utils.values())
        total_util = sum(util_values)
        avg_util = np.mean(util_values) if util_values else 0
        
        # Record result
        result = SolveResult(
            max_utilization=max_util,
            total_utilization=total_util,
            avg_utilization=avg_util,
            latency_term=avg_util,
            solve_time_ms=solve_time_ms,
            solver_status="OPTIMAL",
            latency_weight=latency_weight,
            num_critical_flows=len(critical_flows)
        )
        self.solve_history.append(result)
        
        if self.verbose:
            logger.info(f"Solved in {solve_time_ms:.1f}ms: MLU={max_util:.4f}, avgUtil={avg_util:.4f}")
        
        return max_util, routing
    
    def _compute_link_utilizations(self,
                                    traffic_matrix: np.ndarray,
                                    critical_flows: List[int],
                                    routing: Dict[int, List[float]],
                                    background_routing: str) -> Dict[Tuple[int, int], float]:
        """Compute per-link utilization given routing decisions."""
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
                    if p_idx < len(ratios):
                        for link in self.topo.get_path_links(path):
                            link_load[link] += demand * ratios[p_idx]
            else:
                if background_routing == "single_path":
                    for link in self.topo.get_path_links(paths[0]):
                        link_load[link] += demand
                else:
                    split = demand / len(paths)
                    for path in paths:
                        for link in self.topo.get_path_links(path):
                            link_load[link] += split
        
        utilizations = {}
        for link, load in link_load.items():
            cap = self.topo.get_link_capacity(*link)
            utilizations[link] = load / cap if cap > 0 else 0.0
        
        return utilizations
    
    # =========================================================================
    # BASELINE METHODS (for comparison)
    # =========================================================================
    
    def solve_single_path(self, traffic_matrix: np.ndarray) -> float:
        """All flows use shortest path. Returns MLU."""
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
        """All flows use ECMP (equal split). Returns MLU."""
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
                    latency_weight: float = 0.0) -> Tuple[float, List[int]]:
        """Select top-K flows by demand and optimize them."""
        flow_demands = []
        for flow_idx, (src, dst) in enumerate(self.topo.flow_pairs):
            demand = traffic_matrix[src, dst]
            num_paths = self.topo.get_num_paths(src, dst)
            if num_paths > 1:  # Only multi-path flows
                flow_demands.append((flow_idx, demand))
        
        flow_demands.sort(key=lambda x: -x[1])
        top_k_flows = [f[0] for f in flow_demands[:k]]
        
        util, _ = self.solve(traffic_matrix, top_k_flows, 
                            background_routing="single_path",
                            latency_weight=latency_weight)
        return util, top_k_flows
    
    def solve_random_k(self, traffic_matrix: np.ndarray, k: int,
                       latency_weight: float = 0.0) -> Tuple[float, List[int]]:
        """Select random K flows and optimize them."""
        eligible = []
        for flow_idx, (src, dst) in enumerate(self.topo.flow_pairs):
            demand = traffic_matrix[src, dst]
            if demand <= 0:
                continue
            num_paths = self.topo.get_num_paths(src, dst)
            if num_paths > 1:
                eligible.append(flow_idx)
        
        if len(eligible) <= k:
            selected = eligible
        else:
            selected = list(np.random.choice(eligible, k, replace=False))
        
        util, _ = self.solve(traffic_matrix, selected, 
                            background_routing="single_path",
                            latency_weight=latency_weight)
        return util, selected
    
    # =========================================================================
    # ANALYSIS METHODS
    # =========================================================================
    
    def compare_objectives(self, traffic_matrix: np.ndarray, 
                          critical_flows: List[int],
                          latency_weights: List[float] = [0.0, 0.1, 0.3, 0.5]) -> Dict:
        """
        Compare results across different latency weights.
        Useful for finding the MLU vs latency tradeoff.
        """
        results = {}
        
        for lw in latency_weights:
            util, routing = self.solve(traffic_matrix, critical_flows, latency_weight=lw)
            link_utils = self._compute_link_utilizations(
                traffic_matrix, critical_flows, routing, "single_path")
            
            util_values = list(link_utils.values())
            
            results[lw] = {
                'max_utilization': util,
                'avg_utilization': np.mean(util_values),
                'std_utilization': np.std(util_values),
            }
            
            logger.info(f"latency_weight={lw}: MLU={util:.4f}, avgUtil={results[lw]['avg_utilization']:.4f}")
        
        return results
    
    def get_solve_statistics(self) -> Dict:
        """Get aggregate statistics from all solves."""
        if not self.solve_history:
            return {}
        
        mlu_values = [r.max_utilization for r in self.solve_history]
        times = [r.solve_time_ms for r in self.solve_history]
        
        return {
            'num_solves': len(self.solve_history),
            'avg_mlu': np.mean(mlu_values),
            'avg_solve_time_ms': np.mean(times),
        }


# Alias for backwards compatibility
AbileneLPSolver = AbileneLPSolverLatencyAware


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("LATENCY-AWARE LP SOLVER TEST")
    print("="*60)
    
    topo = AbileneToplogy()
    solver = AbileneLPSolverLatencyAware(topo, verbose=True)
    
    # Create test traffic
    np.random.seed(42)
    tm = np.zeros((topo.num_nodes, topo.num_nodes))
    
    # Some elephant flows
    tm[0, 10] = 2e9   # 2 Gbps
    tm[2, 9] = 1.5e9  # 1.5 Gbps
    tm[1, 8] = 1e9    # 1 Gbps
    
    # Some mice
    for i in range(topo.num_nodes):
        for j in range(topo.num_nodes):
            if i != j and tm[i,j] == 0 and np.random.random() < 0.3:
                tm[i,j] = np.random.uniform(50e6, 200e6)
    
    # Get top-8 flows
    _, top_k = solver.solve_top_k(tm, k=8)
    
    # Compare objectives
    print("\nComparing latency weights:")
    results = solver.compare_objectives(tm, top_k, [0.0, 0.1, 0.3, 0.5])
    
    print("\nSummary:")
    print(f"{'Weight':<10} {'MLU':<10} {'Avg Util':<10}")
    for lw, r in results.items():
        print(f"{lw:<10} {r['max_utilization']:<10.4f} {r['avg_utilization']:<10.4f}")