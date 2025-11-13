package org.cloudbus.cloudsim.sdn.example;

import java.util.ArrayList;
import java.util.Calendar;
import java.util.LinkedList;
import java.util.List;

import org.cloudbus.cloudsim.DatacenterCharacteristics;
import org.cloudbus.cloudsim.Host;
import org.cloudbus.cloudsim.Log;
import org.cloudbus.cloudsim.Storage;
import org.cloudbus.cloudsim.VmAllocationPolicy;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.sdn.HostFactory;
import org.cloudbus.cloudsim.sdn.HostFactorySimple;
import org.cloudbus.cloudsim.sdn.SDNBroker;
import org.cloudbus.cloudsim.sdn.monitor.power.PowerUtilizationMaxHostInterface;
import org.cloudbus.cloudsim.sdn.nos.NetworkOperatingSystem;
import org.cloudbus.cloudsim.sdn.nos.NetworkOperatingSystemSimple;
import org.cloudbus.cloudsim.sdn.parsers.PhysicalTopologyParser;
import org.cloudbus.cloudsim.sdn.policies.selectlink.LinkSelectionPolicy;
import org.cloudbus.cloudsim.sdn.policies.selectlink.LinkSelectionPolicyBandwidthAllocation;
import org.cloudbus.cloudsim.sdn.policies.vmallocation.VmAllocationPolicyCombinedLeastFullFirst;
import org.cloudbus.cloudsim.sdn.rl.RLPipe;
import org.cloudbus.cloudsim.sdn.physicalcomponents.SDNDatacenter;

import org.cloudbus.cloudsim.sdn.rl.RLIntervalEntity;

public class RLExampleSimpleEdge {

    // Use the simple-edge example files
    protected static String physicalFile = "example-edge-simple/edge.physical.json";
    protected static String virtualFile  = "example-edge-simple/edge.virtual.json";

    protected static String[] workloadFiles = {
        "example-edge-simple/edge.workload_host1.csv",
        "example-edge-simple/edge.workload_host2.csv"
    };

    protected static NetworkOperatingSystem nos;
    protected static PowerUtilizationMaxHostInterface maxHostHandler = null;

    public static void main(String[] args) {
        try {
            Log.printLine("Starting RLExampleSimpleEdge...");

            // 1. Initialize CloudSim
            CloudSim.init(1, Calendar.getInstance(), false);

            HostFactory hostFactory = new HostFactorySimple();
            nos = new NetworkOperatingSystemSimple();

            // 2. Load physical topology
            PhysicalTopologyParser.loadPhysicalTopologySingleDC(physicalFile, nos, hostFactory);

            // 3. Set link selection policy
            // Placeholder — this is where the RL policy will be injected later
            LinkSelectionPolicy linkPolicy = new LinkSelectionPolicyBandwidthAllocation();
            nos.setLinkSelectionPolicy(linkPolicy);

            // 4. Create VM allocation policy (simple LFF)
            VmAllocationPolicy vmPolicy =
                new VmAllocationPolicyCombinedLeastFullFirst(nos.getHostList());

            // 5. Create datacenter
            SDNDatacenter datacenter = createDatacenter("Datacenter_0", nos, vmPolicy);

            // 6. Create broker
            SDNBroker broker = new SDNBroker("Broker_0");

            // 7. Submit virtual topology + workloads
            broker.submitDeployApplication(datacenter, virtualFile);
            for (String wf : workloadFiles) {
                broker.submitRequests(wf);
            }

            // START RL INTERVAL ENTITY
            RLPipe pipe = new RLPipe("python");   // assumes 'python' is in PATH
            pipe.startPython();
            RLIntervalEntity rlEntity = new RLIntervalEntity(
                "RL_Entity",
                5.0,   // every 5 seconds
                pipe
            );

            CloudSim.addEntity(rlEntity);


            // 8. Start simulation
            double finishTime = CloudSim.startSimulation();
            CloudSim.stopSimulation();

            //Shutdown RL pipe
            pipe.close();

            Log.printLine("Simulation finished at time: " + finishTime);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static SDNDatacenter createDatacenter(
            String name,
            NetworkOperatingSystem nos,
            VmAllocationPolicy vmPolicy
    ) {
        List<Host> hosts = nos.getHostList();

        String arch = "x86";
        String os = "Linux";
        String vmm = "Xen";

        double time_zone = 10.0;
        double cost = 3.0;
        double costPerMem = 0.05;
        double costPerStorage = 0.001;
        double costPerBw = 0.0;

        LinkedList<Storage> storageList = new LinkedList<Storage>();

        DatacenterCharacteristics characteristics =
            new DatacenterCharacteristics(
                arch, os, vmm, hosts,
                time_zone, cost, costPerMem, costPerStorage, costPerBw
            );

        SDNDatacenter datacenter = null;
        try {
            maxHostHandler = (PowerUtilizationMaxHostInterface) vmPolicy;
            datacenter = new SDNDatacenter(name, characteristics, vmPolicy, storageList, 0, nos);
            nos.setDatacenter(datacenter);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return datacenter;
    }
}
