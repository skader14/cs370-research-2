"""
Fat-Tree Topology for CFR-RL

K=4 Fat-Tree:
- 4 pods, each with 2 edge + 2 aggregation switches
- 4 core switches
- 16 hosts (2 per edge switch)
- Total: 20 switches + 16 hosts = 36 nodes

Path diversity:
- Same edge: 1 path
- Same pod, different edge: 2 paths  
- Different pods: 4 paths (one per core switch)
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Set
import numpy as np
from itertools import product


@dataclass
class FatTreeTopology:
    """K=4 Fat-Tree topology with path enumeration."""
    
    k: int = 4  # Fat-tree parameter (must be even)
    
    def __post_init__(self):
        assert self.k % 2 == 0, "k must be even"
        
        # Counts
        self.num_pods = self.k
        self.num_core = (self.k // 2) ** 2
        self.num_agg_per_pod = self.k // 2
        self.num_edge_per_pod = self.k // 2
        self.num_hosts_per_edge = self.k // 2
        
        self.num_agg = self.num_pods * self.num_agg_per_pod
        self.num_edge = self.num_pods * self.num_edge_per_pod
        self.num_hosts = self.num_edge * self.num_hosts_per_edge
        
        # Node ID ranges
        # Layout: [cores][aggs][edges][hosts]
        self.core_offset = 0
        self.agg_offset = self.num_core
        self.edge_offset = self.agg_offset + self.num_agg
        self.host_offset = self.edge_offset + self.num_edge
        
        self.num_nodes = self.host_offset + self.num_hosts
        self.num_switches = self.num_core + self.num_agg + self.num_edge
        
        # Build topology
        self._build_node_lists()
        self._build_links()
        self._build_path_cache()
        
        # Flow info
        self.num_flows = self.num_hosts * (self.num_hosts - 1)
        self.flow_pairs = [(s, d) for s in self.host_ids for d in self.host_ids if s != d]
        self.flow_to_idx = {pair: i for i, pair in enumerate(self.flow_pairs)}
        
    def _build_node_lists(self):
        """Create node ID lists."""
        self.core_ids = list(range(self.core_offset, self.core_offset + self.num_core))
        self.agg_ids = list(range(self.agg_offset, self.agg_offset + self.num_agg))
        self.edge_ids = list(range(self.edge_offset, self.edge_offset + self.num_edge))
        self.host_ids = list(range(self.host_offset, self.host_offset + self.num_hosts))
        self.switch_ids = self.core_ids + self.agg_ids + self.edge_ids
        
        # Pod membership
        self.pod_of_agg = {}
        self.pod_of_edge = {}
        self.pod_of_host = {}
        
        for pod in range(self.num_pods):
            for i in range(self.num_agg_per_pod):
                agg_id = self.agg_offset + pod * self.num_agg_per_pod + i
                self.pod_of_agg[agg_id] = pod
            for i in range(self.num_edge_per_pod):
                edge_id = self.edge_offset + pod * self.num_edge_per_pod + i
                self.pod_of_edge[edge_id] = pod
                for j in range(self.num_hosts_per_edge):
                    host_id = self.host_offset + (pod * self.num_edge_per_pod + i) * self.num_hosts_per_edge + j
                    self.pod_of_host[host_id] = pod
        
        # Edge switch of each host
        self.edge_of_host = {}
        for edge_idx, edge_id in enumerate(self.edge_ids):
            for j in range(self.num_hosts_per_edge):
                host_id = self.host_offset + edge_idx * self.num_hosts_per_edge + j
                self.edge_of_host[host_id] = edge_id
                
    def _build_links(self):
        """Build adjacency and link capacities."""
        self.adj: Dict[int, List[int]] = {i: [] for i in range(self.num_nodes)}
        self.link_capacity: Dict[Tuple[int, int], float] = {}
        
        # Default capacities (can be asymmetric)
        core_agg_bw = 1e9      # 1 Gbps core-agg links (bottleneck)
        agg_edge_bw = 1e9      # 1 Gbps agg-edge links
        edge_host_bw = 10e9    # 10 Gbps edge-host links (not bottleneck)
        
        # Core to Aggregation links
        # Core switch i connects to aggregation switch (i mod k/2) in each pod
        for core_idx, core_id in enumerate(self.core_ids):
            # Which "column" of agg switches this core connects to
            agg_col = core_idx % (self.k // 2)
            for pod in range(self.num_pods):
                agg_id = self.agg_offset + pod * self.num_agg_per_pod + agg_col
                self._add_link(core_id, agg_id, core_agg_bw)
        
        # Aggregation to Edge links (full mesh within pod)
        for pod in range(self.num_pods):
            for i in range(self.num_agg_per_pod):
                agg_id = self.agg_offset + pod * self.num_agg_per_pod + i
                for j in range(self.num_edge_per_pod):
                    edge_id = self.edge_offset + pod * self.num_edge_per_pod + j
                    self._add_link(agg_id, edge_id, agg_edge_bw)
        
        # Edge to Host links
        for edge_idx, edge_id in enumerate(self.edge_ids):
            for j in range(self.num_hosts_per_edge):
                host_id = self.host_offset + edge_idx * self.num_hosts_per_edge + j
                self._add_link(edge_id, host_id, edge_host_bw)
                
        self.links = list(self.link_capacity.keys())
        self.num_links = len(self.links)
        
    def _add_link(self, u: int, v: int, bw: float):
        """Add bidirectional link."""
        self.adj[u].append(v)
        self.adj[v].append(u)
        self.link_capacity[(u, v)] = bw
        self.link_capacity[(v, u)] = bw
        
    def _build_path_cache(self):
        """Pre-compute all paths between host pairs."""
        self.paths_cache: Dict[Tuple[int, int], List[List[int]]] = {}
        
        for src in self.host_ids:
            for dst in self.host_ids:
                if src != dst:
                    self.paths_cache[(src, dst)] = self._compute_paths(src, dst)
                    
    def _compute_paths(self, src: int, dst: int) -> List[List[int]]:
        """
        Compute all shortest paths between src and dst hosts.
        
        Fat-tree paths:
        - Same edge: H1 -> Edge -> H2 (1 path)
        - Same pod: H1 -> Edge1 -> Agg -> Edge2 -> H2 (k/2 paths, one per agg)
        - Different pods: H1 -> Edge1 -> Agg1 -> Core -> Agg2 -> Edge2 -> H2 (k^2/4 paths)
        """
        src_edge = self.edge_of_host[src]
        dst_edge = self.edge_of_host[dst]
        src_pod = self.pod_of_host[src]
        dst_pod = self.pod_of_host[dst]
        
        paths = []
        
        if src_edge == dst_edge:
            # Same edge switch - single path
            paths.append([src, src_edge, dst])
            
        elif src_pod == dst_pod:
            # Same pod, different edge - route through any agg switch in the pod
            for i in range(self.num_agg_per_pod):
                agg_id = self.agg_offset + src_pod * self.num_agg_per_pod + i
                paths.append([src, src_edge, agg_id, dst_edge, dst])
                
        else:
            # Different pods - route through any (agg, core, agg) combination
            # But fat-tree structure constrains which cores each agg connects to
            for src_agg_idx in range(self.num_agg_per_pod):
                src_agg = self.agg_offset + src_pod * self.num_agg_per_pod + src_agg_idx
                
                # This agg connects to cores in positions [src_agg_idx * k/2, (src_agg_idx+1) * k/2)
                # Actually, agg at position i in pod connects to cores i, i + k/2, i + k, ...
                # Let me reconsider: core j connects to agg (j mod k/2) in each pod
                # So agg at position i connects to cores i, i + k/2 (for k=4: positions 0,2 or 1,3)
                
                for core_offset in range(self.k // 2):
                    core_idx = src_agg_idx + core_offset * (self.k // 2)
                    if core_idx < self.num_core:
                        core_id = self.core_ids[core_idx]
                        
                        # Find dst_agg that connects to this core
                        dst_agg_idx = core_idx % (self.k // 2)
                        dst_agg = self.agg_offset + dst_pod * self.num_agg_per_pod + dst_agg_idx
                        
                        paths.append([src, src_edge, src_agg, core_id, dst_agg, dst_edge, dst])
        
        return paths
    
    def get_all_paths(self, src: int, dst: int) -> List[List[int]]:
        """Return all paths between src and dst."""
        return self.paths_cache.get((src, dst), [])
    
    def get_num_paths(self, src: int, dst: int) -> int:
        """Return number of paths between src and dst."""
        return len(self.get_all_paths(src, dst))
    
    def get_path_links(self, path: List[int]) -> List[Tuple[int, int]]:
        """Convert path (list of nodes) to list of links."""
        return [(path[i], path[i+1]) for i in range(len(path) - 1)]
    
    def get_link_capacity(self, u: int, v: int) -> float:
        """Get capacity of link (u, v)."""
        return self.link_capacity.get((u, v), 0.0)
    
    def hosts_same_edge(self, h1: int, h2: int) -> bool:
        """Check if two hosts share the same edge switch."""
        return self.edge_of_host[h1] == self.edge_of_host[h2]
    
    def hosts_same_pod(self, h1: int, h2: int) -> bool:
        """Check if two hosts are in the same pod."""
        return self.pod_of_host[h1] == self.pod_of_host[h2]
    
    def get_flow_type(self, src: int, dst: int) -> str:
        """Classify flow by path diversity."""
        if self.hosts_same_edge(src, dst):
            return "same_edge"
        elif self.hosts_same_pod(src, dst):
            return "same_pod"
        else:
            return "cross_pod"
    
    def print_summary(self):
        """Print topology summary."""
        print(f"Fat-Tree Topology (k={self.k})")
        print(f"  Pods: {self.num_pods}")
        print(f"  Core switches: {self.num_core}")
        print(f"  Aggregation switches: {self.num_agg} ({self.num_agg_per_pod} per pod)")
        print(f"  Edge switches: {self.num_edge} ({self.num_edge_per_pod} per pod)")
        print(f"  Hosts: {self.num_hosts} ({self.num_hosts_per_edge} per edge)")
        print(f"  Total nodes: {self.num_nodes}")
        print(f"  Total links: {self.num_links}")
        print(f"  Total flows: {self.num_flows}")
        
        # Count flows by type
        same_edge = sum(1 for s, d in self.flow_pairs if self.get_flow_type(s, d) == "same_edge")
        same_pod = sum(1 for s, d in self.flow_pairs if self.get_flow_type(s, d) == "same_pod")
        cross_pod = sum(1 for s, d in self.flow_pairs if self.get_flow_type(s, d) == "cross_pod")
        
        print(f"\nFlow breakdown:")
        print(f"  Same edge (1 path): {same_edge} flows")
        print(f"  Same pod (2 paths): {same_pod} flows")
        print(f"  Cross pod (4 paths): {cross_pod} flows")
        
        # Sample path counts
        print(f"\nSample paths:")
        samples = [
            (self.host_ids[0], self.host_ids[1]),   # Same edge
            (self.host_ids[0], self.host_ids[2]),   # Same pod
            (self.host_ids[0], self.host_ids[4]),   # Cross pod
        ]
        for src, dst in samples:
            paths = self.get_all_paths(src, dst)
            ftype = self.get_flow_type(src, dst)
            print(f"  {src} -> {dst} ({ftype}): {len(paths)} paths")
            if paths:
                print(f"    Example: {paths[0]}")


def test_fat_tree():
    """Test fat-tree topology."""
    topo = FatTreeTopology(k=4)
    topo.print_summary()
    
    print("\n" + "="*60)
    print("PATH VERIFICATION")
    print("="*60)
    
    # Test all flow types
    h = topo.host_ids
    
    # Same edge: hosts 0 and 1 share edge switch 0
    print(f"\nSame edge ({h[0]} -> {h[1]}):")
    for path in topo.get_all_paths(h[0], h[1]):
        print(f"  {path}")
    
    # Same pod: hosts 0 and 2 in same pod but different edge
    print(f"\nSame pod ({h[0]} -> {h[2]}):")
    for path in topo.get_all_paths(h[0], h[2]):
        print(f"  {path}")
    
    # Cross pod: hosts 0 and 4 in different pods
    print(f"\nCross pod ({h[0]} -> {h[4]}):")
    for path in topo.get_all_paths(h[0], h[4]):
        print(f"  {path}")
        
    # Verify link structure
    print("\n" + "="*60)
    print("LINK STRUCTURE")
    print("="*60)
    
    print(f"\nCore switch connections:")
    for core_id in topo.core_ids[:2]:  # First 2 cores
        neighbors = topo.adj[core_id]
        print(f"  Core {core_id}: connects to aggs {neighbors}")
        
    print(f"\nAgg switch connections (Pod 0):")
    for i in range(topo.num_agg_per_pod):
        agg_id = topo.agg_ids[i]
        neighbors = topo.adj[agg_id]
        cores = [n for n in neighbors if n in topo.core_ids]
        edges = [n for n in neighbors if n in topo.edge_ids]
        print(f"  Agg {agg_id}: cores={cores}, edges={edges}")


if __name__ == "__main__":
    test_fat_tree()