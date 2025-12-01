"""
Network topology representation for LP solver.
Multi-path topology: 2 core switches, N edge switches, hosts per edge.
This creates path diversity needed for traffic engineering.
"""
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import itertools

class Topology:
    """
    Dual-core tree topology for datacenter network.
    
    Structure:
        - 2 core switches (nodes 0 and 1) - creates path redundancy
        - N_EDGE edge switches (nodes 2 to N_EDGE+1)  
        - HOSTS_PER_EDGE hosts under each edge switch
        
    Path diversity:
        - Same edge: 1 path (direct)
        - Different edge: 2 paths (via core0 or core1)
    """
    
    def __init__(self, n_edge: int = 4, hosts_per_edge: int = 4, 
             core_bw: float = 1e9, edge_bw: float = 10e9): 
        """
        Args:
            n_edge: Number of edge switches
            hosts_per_edge: Hosts per edge switch
            core_bw: Bandwidth of core-edge links (bps)
            edge_bw: Bandwidth of edge-host links (bps)
        """
        self.n_edge = n_edge
        self.hosts_per_edge = hosts_per_edge
        self.n_hosts = n_edge * hosts_per_edge
        self.n_cores = 2  # Dual core for path diversity
        self.core_bw = core_bw
        self.edge_bw = edge_bw
        
        # Node IDs
        # 0, 1 = core switches
        # 2 to n_edge+1 = edge switches
        # n_edge+2 onwards = hosts
        self.core_ids = [0, 1]
        self.edge_ids = list(range(2, 2 + n_edge))
        self.host_ids = list(range(2 + n_edge, 2 + n_edge + self.n_hosts))
        
        self.n_nodes = 2 + n_edge + self.n_hosts
        
        # Build adjacency and link info
        self._build_topology()
        
    def _build_topology(self):
        """Build adjacency list and link capacities."""
        self.adj: Dict[int, List[int]] = defaultdict(list)
        self.link_capacity: Dict[Tuple[int, int], float] = {}
        self.links: List[Tuple[int, int]] = []
        
        # Core to edge links (each edge connects to BOTH cores)
        for core_id in self.core_ids:
            for edge_id in self.edge_ids:
                self.adj[core_id].append(edge_id)
                self.adj[edge_id].append(core_id)
                
                self.link_capacity[(core_id, edge_id)] = self.core_bw
                self.link_capacity[(edge_id, core_id)] = self.core_bw
                self.links.append((core_id, edge_id))
                self.links.append((edge_id, core_id))
        
        # Edge to host links
        for edge_idx, edge_id in enumerate(self.edge_ids):
            for h in range(self.hosts_per_edge):
                host_id = self.host_id_from_edge(edge_idx, h)
                self.adj[edge_id].append(host_id)
                self.adj[host_id].append(edge_id)
                
                self.link_capacity[(edge_id, host_id)] = self.edge_bw
                self.link_capacity[(host_id, edge_id)] = self.edge_bw
                self.links.append((edge_id, host_id))
                self.links.append((host_id, edge_id))
        
        # Map host_id to its edge switch
        self.host_to_edge: Dict[int, int] = {}
        for edge_idx, edge_id in enumerate(self.edge_ids):
            for h in range(self.hosts_per_edge):
                host_id = self.host_id_from_edge(edge_idx, h)
                self.host_to_edge[host_id] = edge_id
                
    def host_id_from_edge(self, edge_idx: int, host_idx: int) -> int:
        """Get node ID for host given edge index and local host index."""
        return 2 + self.n_edge + edge_idx * self.hosts_per_edge + host_idx
    
    def get_host_edge(self, host_id: int) -> int:
        """Get edge switch ID for a host."""
        return self.host_to_edge[host_id]
    
    def get_all_paths(self, src_host: int, dst_host: int) -> List[List[int]]:
        """
        Get all paths between two hosts.
        
        Returns:
            List of paths, where each path is a list of node IDs
        """
        if src_host == dst_host:
            return [[src_host]]
            
        src_edge = self.host_to_edge[src_host]
        dst_edge = self.host_to_edge[dst_host]
        
        if src_edge == dst_edge:
            # Same edge switch - single path
            return [[src_host, src_edge, dst_host]]
        else:
            # Different edge switches - two paths (via each core)
            paths = []
            for core_id in self.core_ids:
                path = [src_host, src_edge, core_id, dst_edge, dst_host]
                paths.append(path)
            return paths
    
    def get_path_links(self, path: List[int]) -> List[Tuple[int, int]]:
        """Convert path (list of nodes) to list of directed links."""
        return [(path[i], path[i+1]) for i in range(len(path) - 1)]
    
    def get_all_flows(self) -> List[Tuple[int, int]]:
        """Get all possible (src, dst) host pairs."""
        return [(s, d) for s in self.host_ids for d in self.host_ids if s != d]
    
    def get_n_flows(self) -> int:
        """Total number of possible flows."""
        return self.n_hosts * (self.n_hosts - 1)
    
    def hosts_same_edge(self, host1: int, host2: int) -> bool:
        """Check if two hosts are under the same edge switch."""
        return self.host_to_edge[host1] == self.host_to_edge[host2]


def test_topology():
    """Quick sanity check."""
    topo = Topology(n_edge=4, hosts_per_edge=4)
    print(f"Nodes: {topo.n_nodes}")
    print(f"Hosts: {topo.n_hosts}")
    print(f"Core switches: {topo.core_ids}")
    print(f"Edge switches: {topo.edge_ids}")
    print(f"Links: {len(topo.links)}")
    print(f"Flows: {topo.get_n_flows()}")
    
    # Test paths - same edge
    h1 = topo.host_ids[0]  # First host (edge 0)
    h2 = topo.host_ids[1]  # Second host (edge 0)
    paths = topo.get_all_paths(h1, h2)
    print(f"\nSame edge - Path {h1} -> {h2}: {paths}")
    
    # Test paths - different edge
    h3 = topo.host_ids[-1]  # Last host (edge 3)
    paths = topo.get_all_paths(h1, h3)
    print(f"Different edge - Paths {h1} -> {h3}:")
    for i, p in enumerate(paths):
        print(f"  Path {i+1}: {p} via core {topo.core_ids[i]}")
        print(f"    Links: {topo.get_path_links(p)}")


if __name__ == "__main__":
    test_topology()