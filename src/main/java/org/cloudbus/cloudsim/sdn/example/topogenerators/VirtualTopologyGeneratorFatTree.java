package org.cloudbus.cloudsim.sdn.example.topogenerators;

import java.io.FileWriter;
import java.io.IOException;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;

public class VirtualTopologyGeneratorFatTree {
    
    @SuppressWarnings("unchecked")
    public void generate(int numPods, String filename) {
        JSONObject topology = new JSONObject();
        JSONArray nodes = new JSONArray();
        JSONArray links = new JSONArray();
        
        int numHosts = (numPods * numPods * numPods) / 4;
        
        // Create VMs - one per host, using fat-tree naming
        int vmId = 0;
        for (int pod = 0; pod < numPods; pod++) {
            for (int edge = 0; edge < numPods/2; edge++) {
                for (int h = 0; h < numPods/2; h++) {
                    JSONObject vm = new JSONObject();
                    vm.put("name", "vm_" + vmId);
                    vm.put("type", "vm");
                    vm.put("size", 1000L);
                    vm.put("pes", 1);
                    vm.put("mips", 10000L);
                    vm.put("ram", 512);
                    vm.put("bw", 1000000000L);
                    vm.put("host", "h_" + pod + "_" + edge + "_" + h);
                    nodes.add(vm);
                    vmId++;
                }
            }
        }
        
        // Create flows - all pairs
        int flowId = 0;
        for (int src = 0; src < numHosts; src++) {
            for (int dst = 0; dst < numHosts; dst++) {
                if (src != dst) {
                    JSONObject link = new JSONObject();
                    link.put("name", "flow_" + flowId);
                    link.put("source", "vm_" + src);
                    link.put("destination", "vm_" + dst);
                    link.put("bandwidth", 1000000L);
                    links.add(link);
                    flowId++;
                }
            }
        }
        
        topology.put("nodes", nodes);
        topology.put("links", links);
        
        try (FileWriter file = new FileWriter(filename)) {
            file.write(topology.toJSONString().replaceAll(",", ",\n"));
        } catch (IOException e) {
            e.printStackTrace();
        }
        
        System.out.println("Generated " + vmId + " VMs and " + flowId + " flows");
    }
}