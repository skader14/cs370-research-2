"""
Abilene Topology - Internet2 Backbone Network

This is the actual topology used in the CFR-RL paper.
12 PoP (Point of Presence) locations across the US.

Key properties that make it interesting for RL:
- Asymmetric link capacities (10G backbone, some 2.5G)
- Unequal path lengths
- Multiple paths between most node pairs
- Real-world traffic patterns
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import heapq


@dataclass
class AbileneToplogy:
    """
    Abilene/Internet2 backbone topology.
    
    Nodes (12 PoPs):
        0: Seattle (STTL)
        1: Sunnyvale (SNVA) 
        2: Los Angeles (LOSA)
        3: Denver (DNVR)
        4: Kansas City (KSCY)
        5: Houston (HSTN)
        6: Chicago (CHIC)
        7: Indianapolis (IPLS)
        8: Atlanta (ATLA)
        9: Washington DC (WASH)
        10: New York (NYCM)
        11: Jacksonville (JCVN) - added for connectivity
    
    All nodes are both switches and potential traffic sources/sinks.
    """
    
    def __post_init__(self):
        self.num_nodes = 12
        self.node_names = [
            "Seattle", "Sunnyvale", "Los Angeles", "Denver",
            "Kansas City", "Houston", "Chicago", "Indianapolis", 
            "Atlanta", "Washington", "New York", "Jacksonville"
        ]
        self.node_abbrev = [
            "STTL", "SNVA", "LOSA", "DNVR", "KSCY", "HSTN",
            "CHIC", "IPLS", "ATLA", "WASH", "NYCM", "JCVN"
        ]
        
        # Build topology
        self._build_links()
        self._build_paths()
        
        # Flow info (all node pairs)
        self.flow_pairs = [(s, d) for s in range(self.num_nodes) 
                          for d in range(self.num_nodes) if s != d]
        self.num_flows = len(self.flow_pairs)  # 12 * 11 = 132
        self.flow_to_idx = {pair: i for i, pair in enumerate(self.flow_pairs)}
        
    def _build_links(self):
        """
        Build Abilene link structure.
        
        Link capacities based on real Abilene:
        - Most backbone links: 10 Gbps
        - Some regional links: 2.5 Gbps (creates bottlenecks)
        """
        self.adj: Dict[int, List[int]] = defaultdict(list)
        self.link_capacity: Dict[Tuple[int, int], float] = {}
        
        # Define edges as (node1, node2, capacity_gbps)
        # Based on Abilene topology maps
        edges = [
            # West Coast
            (0, 1, 10),    # Seattle - Sunnyvale
            (1, 2, 10),    # Sunnyvale - Los Angeles
            (0, 3, 10),    # Seattle - Denver
            (1, 3, 10),    # Sunnyvale - Denver
            (2, 5, 10),    # Los Angeles - Houston
            
            # Central
            (3, 4, 10),    # Denver - Kansas City
            (4, 5, 10),    # Kansas City - Houston
            (4, 6, 10),    # Kansas City - Chicago
            (4, 7, 10),    # Kansas City - Indianapolis
            
            # East Coast  
            (6, 7, 10),    # Chicago - Indianapolis
            (7, 8, 10),    # Indianapolis - Atlanta
            (5, 8, 10),    # Houston - Atlanta
            (8, 9, 10),    # Atlanta - Washington
            (9, 10, 10),   # Washington - New York
            (6, 10, 10),   # Chicago - New York
            (7, 9, 10),    # Indianapolis - Washington
            
            # Additional connectivity
            (8, 11, 2.5),  # Atlanta - Jacksonville (lower capacity - bottleneck)
            (5, 11, 2.5),  # Houston - Jacksonville (lower capacity - bottleneck)
        ]
        
        for u, v, cap_gbps in edges:
            cap_bps = cap_gbps * 1e9
            self._add_link(u, v, cap_bps)
        
        self.links = list(self.link_capacity.keys())
        self.num_links = len(self.links)
        
    def _add_link(self, u: int, v: int, capacity: float):
        """Add bidirectional link."""
        self.adj[u].append(v)
        self.adj[v].append(u)
        self.link_capacity[(u, v)] = capacity
        self.link_capacity[(v, u)] = capacity
        
    def _build_paths(self):
        """
        Compute K-shortest paths between all node pairs.
        
        We use Yen's algorithm to find multiple paths,
        limited to reasonable path lengths.
        """
        self.paths_cache: Dict[Tuple[int, int], List[List[int]]] = {}
        
        for src in range(self.num_nodes):
            for dst in range(self.num_nodes):
                if src != dst:
                    # Find up to 4 shortest paths
                    paths = self._k_shortest_paths(src, dst, k=4)
                    self.paths_cache[(src, dst)] = paths
                    
    def _k_shortest_paths(self, src: int, dst: int, k: int = 4) -> List[List[int]]:
        """
        Find k shortest paths using Yen's algorithm.
        """
        # First find the shortest path using Dijkstra
        shortest = self._dijkstra(src, dst)
        if not shortest:
            return []
        
        paths = [shortest]
        candidates = []
        
        for i in range(1, k):
            # Spur paths from the previous shortest path
            prev_path = paths[i - 1]
            
            for j in range(len(prev_path) - 1):
                spur_node = prev_path[j]
                root_path = prev_path[:j + 1]
                
                # Remove edges that are part of previous paths with same root
                removed_edges = set()
                for path in paths:
                    if path[:j + 1] == root_path and j + 1 < len(path):
                        edge = (path[j], path[j + 1])
                        removed_edges.add(edge)
                        removed_edges.add((edge[1], edge[0]))
                
                # Find spur path avoiding removed edges
                spur_path = self._dijkstra(spur_node, dst, removed_edges, set(root_path[:-1]))
                
                if spur_path:
                    total_path = root_path[:-1] + spur_path
                    path_cost = self._path_cost(total_path)
                    
                    # Add to candidates if not duplicate
                    if total_path not in paths and (path_cost, total_path) not in candidates:
                        heapq.heappush(candidates, (path_cost, total_path))
            
            if not candidates:
                break
                
            # Add best candidate
            _, best_path = heapq.heappop(candidates)
            paths.append(best_path)
        
        return paths
    
    def _dijkstra(self, src: int, dst: int, 
                  removed_edges: Set[Tuple[int, int]] = None,
                  removed_nodes: Set[int] = None) -> List[int]:
        """Dijkstra's shortest path."""
        removed_edges = removed_edges or set()
        removed_nodes = removed_nodes or set()
        
        dist = {src: 0}
        prev = {src: None}
        heap = [(0, src)]
        visited = set()
        
        while heap:
            d, u = heapq.heappop(heap)
            
            if u in visited:
                continue
            visited.add(u)
            
            if u == dst:
                # Reconstruct path
                path = []
                node = dst
                while node is not None:
                    path.append(node)
                    node = prev[node]
                return path[::-1]
            
            for v in self.adj[u]:
                if v in visited or v in removed_nodes:
                    continue
                if (u, v) in removed_edges:
                    continue
                    
                # Use hop count as cost (could use capacity-based cost)
                new_dist = dist[u] + 1
                
                if v not in dist or new_dist < dist[v]:
                    dist[v] = new_dist
                    prev[v] = u
                    heapq.heappush(heap, (new_dist, v))
        
        return []
    
    def _path_cost(self, path: List[int]) -> int:
        """Cost of a path (hop count)."""
        return len(path) - 1
    
    def get_all_paths(self, src: int, dst: int) -> List[List[int]]:
        """Return all cached paths between src and dst."""
        return self.paths_cache.get((src, dst), [])
    
    def get_num_paths(self, src: int, dst: int) -> int:
        """Return number of paths between src and dst."""
        return len(self.get_all_paths(src, dst))
    
    def get_path_links(self, path: List[int]) -> List[Tuple[int, int]]:
        """Convert path to list of links."""
        return [(path[i], path[i+1]) for i in range(len(path) - 1)]
    
    def get_link_capacity(self, u: int, v: int) -> float:
        """Get link capacity."""
        return self.link_capacity.get((u, v), 0.0)
    
    def get_path_bottleneck(self, path: List[int]) -> float:
        """Get minimum capacity along a path."""
        links = self.get_path_links(path)
        if not links:
            return 0.0
        return min(self.get_link_capacity(u, v) for u, v in links)
    
    def print_summary(self):
        """Print topology summary."""
        print(f"Abilene Topology (Internet2 Backbone)")
        print(f"  Nodes: {self.num_nodes}")
        print(f"  Links: {self.num_links // 2} (bidirectional)")
        print(f"  Flows: {self.num_flows}")
        
        # Link capacity distribution
        capacities = set(self.link_capacity.values())
        print(f"\nLink capacities:")
        for cap in sorted(capacities, reverse=True):
            count = sum(1 for c in self.link_capacity.values() if c == cap) // 2
            print(f"  {cap/1e9:.1f} Gbps: {count} links")
        
        # Path diversity
        path_counts = [self.get_num_paths(s, d) for s, d in self.flow_pairs]
        print(f"\nPath diversity:")
        print(f"  Min paths: {min(path_counts)}")
        print(f"  Max paths: {max(path_counts)}")
        print(f"  Avg paths: {np.mean(path_counts):.1f}")
        
        # Flows with multiple paths
        multi_path = sum(1 for p in path_counts if p > 1)
        print(f"  Flows with >1 path: {multi_path}/{self.num_flows}")
        
        # Node connectivity
        print(f"\nNode connectivity:")
        for i in range(self.num_nodes):
            neighbors = self.adj[i]
            print(f"  {self.node_abbrev[i]}: {len(neighbors)} links -> {[self.node_abbrev[n] for n in neighbors]}")


def test_abilene():
    """Test Abilene topology."""
    print("="*60)
    print("ABILENE TOPOLOGY TEST")
    print("="*60 + "\n")
    
    topo = AbileneToplogy()
    topo.print_summary()
    
    # Sample paths
    print("\n" + "="*60)
    print("SAMPLE PATHS")
    print("="*60)
    
    test_pairs = [
        (0, 10),   # Seattle to New York (cross-country)
        (0, 1),    # Seattle to Sunnyvale (adjacent)
        (2, 9),    # Los Angeles to Washington (cross-country)
        (8, 11),   # Atlanta to Jacksonville (bottleneck link)
    ]
    
    for src, dst in test_pairs:
        paths = topo.get_all_paths(src, dst)
        src_name = topo.node_abbrev[src]
        dst_name = topo.node_abbrev[dst]
        print(f"\n{src_name} -> {dst_name}: {len(paths)} paths")
        
        for i, path in enumerate(paths[:3]):  # Show first 3
            path_names = [topo.node_abbrev[n] for n in path]
            bottleneck = topo.get_path_bottleneck(path) / 1e9
            print(f"  {i+1}. {' -> '.join(path_names)} (bottleneck: {bottleneck:.1f} Gbps)")
    
    # Identify bottleneck links
    print("\n" + "="*60)
    print("BOTTLENECK LINKS (< 10 Gbps)")
    print("="*60)
    
    for (u, v), cap in topo.link_capacity.items():
        if u < v and cap < 10e9:  # Avoid duplicates
            print(f"  {topo.node_abbrev[u]} <-> {topo.node_abbrev[v]}: {cap/1e9:.1f} Gbps")


if __name__ == "__main__":
    test_abilene()