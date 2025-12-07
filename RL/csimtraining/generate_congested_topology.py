"""
Generate a Fat-Tree topology guaranteed to cause congestion.

This version sets BOTH link bandwidth AND flow bandwidth to very low values.

The key insight: CloudSim may use FLOW bandwidth (from virtual.json) 
for transmission time, not LINK bandwidth (from physical.json).

Usage:
    python generate_congested_topology.py
"""

import json
from pathlib import Path

def generate_congested_fattree(output_dir: str = "dataset-fattree"):
    """
    Generate topology with guaranteed congestion.
    
    Settings:
    - Link bandwidth: 10 Mbps (10,000,000 bps)
    - Flow bandwidth: 100 Kbps (100,000 bps) <-- KEY CHANGE
    
    At 100 Kbps per flow:
    - 500 KB packet = 4,000,000 bits
    - Transmission time = 4,000,000 / 100,000 = 40 seconds!
    - This WILL cause timeout and drops
    """
    
    k = 4  # Fat-tree parameter
    
    # Bandwidth settings - VERY LOW to guarantee congestion
    link_bw = 10_000_000      # 10 Mbps for links
    flow_bw = 100_000         # 100 Kbps per flow (KEY!)
    latency = 0.001           # 1ms
    
    print("=" * 60)
    print("GENERATING CONGESTED FAT-TREE TOPOLOGY")
    print("=" * 60)
    print(f"  Link bandwidth: {link_bw/1e6:.0f} Mbps")
    print(f"  Flow bandwidth: {flow_bw/1e3:.0f} Kbps  <-- LOW for congestion")
    print()
    
    # Calculate expected transmission times
    pkt_sizes = [100_000, 500_000, 1_000_000]  # 100KB, 500KB, 1MB
    print("  Expected transmission times at flow bandwidth:")
    for size in pkt_sizes:
        bits = size * 8
        tx_time = bits / flow_bw
        print(f"    {size/1000:.0f} KB packet: {tx_time:.1f} seconds")
    print()
    
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
                "bw": link_bw
            })
    
    # Pods
    for pod in range(k):
        edge_switches = []
        agg_switches = []
        
        for i in range(k // 2):
            e_name = f"e_{pod}_{i}"
            a_name = f"a_{pod}_{i}"
            
            edge_switches.append(e_name)
            agg_switches.append(a_name)
            
            nodes.append({"name": e_name, "type": "edge", "iops": sw_iops, "bw": link_bw})
            nodes.append({"name": a_name, "type": "aggregate", "iops": sw_iops, "bw": link_bw})
            
            links.append({"source": a_name, "destination": e_name, "latency": latency})
            
            for j in range(i):
                links.append({"source": agg_switches[i], "destination": edge_switches[j], "latency": latency})
                links.append({"source": agg_switches[j], "destination": edge_switches[i], "latency": latency})
            
            for j in range(k // 2):
                links.append({"source": a_name, "destination": core_switches[(i, j)], "latency": latency})
            
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
                    "bw": link_bw
                })
                links.append({"source": e_name, "destination": h_name, "latency": latency})
    
    # Save physical topology
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    physical = {"nodes": nodes, "links": links}
    with open(f"{output_dir}/fattree-physical.json", 'w') as f:
        json.dump(physical, f, indent=2)
    print(f"  Saved: {output_dir}/fattree-physical.json")
    
    # Generate virtual topology with LOW flow bandwidth
    num_hosts = len(hosts_list)
    vm_nodes = []
    flow_links = []
    
    for vm_id, host_name in enumerate(hosts_list):
        vm_nodes.append({
            "name": f"vm_{vm_id}",
            "type": "vm",
            "size": 1000,
            "pes": 1,
            "mips": 10000,
            "ram": 512,
            "bw": link_bw,  # VM bandwidth
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
                    "bandwidth": flow_bw  # LOW bandwidth per flow
                })
                flow_id += 1
    
    virtual = {"nodes": vm_nodes, "links": flow_links}
    with open(f"{output_dir}/fattree-virtual.json", 'w') as f:
        json.dump(virtual, f, indent=2)
    print(f"  Saved: {output_dir}/fattree-virtual.json")
    
    # Summary
    print()
    print(f"  Topology: Fat-Tree k={k}")
    print(f"  Hosts: {num_hosts}")
    print(f"  Switches: {len([n for n in nodes if n['type'] != 'host'])}")
    print(f"  Links: {len(links)}")
    print(f"  Flows: {flow_id}")
    print()
    print("  With 300 packets of ~500KB each:")
    print("    - Each packet takes ~40 seconds to transmit")
    print("    - 10 second episode timeout")
    print("    - Packets WILL drop!")
    print()
    print("=" * 60)


if __name__ == "__main__":
    generate_congested_fattree()