package org.cloudbus.cloudsim.sdn.example.topogenerators;

public class GenerateFatTree {
    public static void main(String[] args) {
        String physicalFile = "dataset-fattree/fattree-physical.json";
        String virtualFile = "dataset-fattree/fattree-virtual.json";
        
        int numPods = 4;  // k=4 fat-tree: 16 hosts
        double latency = 0.001;  // 1ms
        long iops = 1000000000L;
        long bw = 1000000000L;  // 1 Gbps
        
        // Host specs
        int pe = 8;
        long mips = 10000;
        int ram = 16384;
        long storage = 1000000;
        
        // Generate physical topology
        PhysicalTopologyGenerator physGen = new PhysicalTopologyGenerator();
        PhysicalTopologyGenerator.HostSpec hostSpec = physGen.createHostSpec(pe, mips, ram, storage, bw);
        physGen.createTopologyFatTree(hostSpec, iops, bw, numPods, latency);
        physGen.wrtieJSON(physicalFile);
        
        System.out.println("Generated: " + physicalFile);
        
        // Generate virtual topology  
        VirtualTopologyGeneratorFatTree virtGen = new VirtualTopologyGeneratorFatTree();
        virtGen.generate(numPods, virtualFile);
        
        System.out.println("Generated: " + virtualFile);
    }
}