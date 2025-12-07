"""
Generate a Fat-Tree k=4 topology with reduced bandwidth.

With 1 Gbps links, we never see congestion. 
10 Mbps creates realistic bottlenecks while maintaining full path diversity.

Usage:
    python generate_low_bw_topology.py --bandwidth 10  # 10 Mbps (recommended)
"""

import json
import argparse
from pathlib import Path

def generate_fattree_topology(k: int = 4, bandwidth_mbps: int = 10, output_dir: str = "dataset-fattree"):
    """
    Generate Fat-Tree physical and virtual topology files.
    
    Args:
        k: Fat-tree parameter (k=4 gives 16 hosts)
        bandwidth_mbps: Link bandwidth in Mbps (10 = 10 Mbps)
        output_dir: Output directory for JSON files
    """
    bw = bandwidth_mbps * 1_000_000  # Convert to bps
    latency = 0.001  # 1ms
    
    # Switch/Host specs
    sw_iops = 1_000_000_000
    host_pe = 8
    host_mips = 10000
    host_ram = 16384
    host_storage = 1_000_000
    
    nodes = []
    links = []
    hosts_list = []
    
    # Core switches
    core_switches = {}
    for i in range(k // 2):
        for j in range(k // 2):
            name = f"c_{i}_{j}"
            core_switches[(i, j)] = name
            nodes.append({
                "name": name,
                "type": "core",
                "iops": sw_iops,
                "bw": bw
            })
    
    # For each pod
    for pod in range(k):
        edge_switches = []
        agg_switches = []
        
        for i in range(k // 2):
            e_name = f"e_{pod}_{i}"
            a_name = f"a_{pod}_{i}"
            
            edge_switches.append(e_name)
            agg_switches.append(a_name)
            
            nodes.append({"name": e_name, "type": "edge", "iops": sw_iops, "bw": bw})
            nodes.append({"name": a_name, "type": "aggregate", "iops": sw_iops, "bw": bw})
            
            # Aggregate to edge links
            links.append({"source": a_name, "destination": e_name, "latency": latency})
            
            # Cross-links within pod
            for j in range(i):
                links.append({"source": agg_switches[i], "destination": edge_switches[j], "latency": latency})
                links.append({"source": agg_switches[j], "destination": edge_switches[i], "latency": latency})
            
            # Aggregate to core links
            for j in range(k // 2):
                links.append({"source": a_name, "destination": core_switches[(i, j)], "latency": latency})
            
            # Hosts under edge switch
            for j in range(k // 2):
                h_name = f"h_{pod}_{i}_{j}"
                hosts_list.append(h_name)
                nodes.append({
                    "name": h_name,
                    "type": "host",
                    "pes": host_pe,
                    "mips": host_mips,
                    "ram": host_ram,
                    "storage": host_storage,
                    "bw": bw
                })
                links.append({"source": e_name, "destination": h_name, "latency": latency})
    
    # Save physical topology
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    physical = {"nodes": nodes, "links": links}
    with open(f"{output_dir}/fattree-physical.json", 'w') as f:
        json.dump(physical, f, indent=2)
    
    # Generate virtual topology
    num_hosts = len(hosts_list)
    vm_nodes = []
    flow_links = []
    
    # Flow bandwidth = link_bw / 10 (so 10 concurrent flows saturate a link)
    flow_bw = bw // 10
    
    for vm_id, host_name in enumerate(hosts_list):
        vm_nodes.append({
            "name": f"vm_{vm_id}",
            "type": "vm",
            "size": 1000,
            "pes": 1,
            "mips": 10000,
            "ram": 512,
            "bw": bw,
            "host": host_name
        })
    
    flow_id = 0
    for src in range(num_hosts):
        for dst in range(num_hosts):
            if src != dst:
                flow_links.append({
                    "name": f"flow_{flow_id}",
                    "source": f"vm_{src}",
                    "destination": f"vm_{dst}",
                    "bandwidth": flow_bw
                })
                flow_id += 1
    
    virtual = {"nodes": vm_nodes, "links": flow_links}
    with open(f"{output_dir}/fattree-virtual.json", 'w') as f:
        json.dump(virtual, f, indent=2)
    
    print(f"Generated Fat-Tree k={k} topology with {bandwidth_mbps} Mbps links:")
    print(f"  Hosts: {num_hosts}")
    print(f"  Switches: {len([n for n in nodes if n['type'] != 'host'])}")
    print(f"  Links: {len(links)}")
    print(f"  Flows: {flow_id}")
    print(f"  Flow bandwidth: {flow_bw/1e6:.1f} Mbps")
    print(f"  Output: {output_dir}/")
    
    # Calculate expected congestion with 300 packets
    print(f"\n  === Expected Utilization (300 packets, 10s) ===")
    avg_pkt_kb = 500  # ~500 KB average
    total_bytes = 300 * avg_pkt_kb * 1000
    rate_mbps = total_bytes * 8 / 10 / 1e6
    edge_capacity = bandwidth_mbps * num_hosts
    print(f"  Traffic rate: ~{rate_mbps:.0f} Mbps")
    print(f"  Edge capacity: {edge_capacity} Mbps")
    print(f"  Expected utilization: ~{rate_mbps/edge_capacity*100:.0f}%")
    
    if rate_mbps / edge_capacity > 0.3:
        print(f"  ✓ Should see congestion!")
    else:
        print(f"  ✗ May need more traffic")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Fat-Tree topology with configurable bandwidth")
    parser.add_argument("--bandwidth", type=int, default=10, help="Link bandwidth in Mbps (default: 10)")
    parser.add_argument("--output-dir", type=str, default="dataset-fattree", help="Output directory")
    parser.add_argument("-k", type=int, default=4, help="Fat-tree k parameter (default: 4)")
    
    args = parser.parse_args()
    generate_fattree_topology(k=args.k, bandwidth_mbps=args.bandwidth, output_dir=args.output_dir)