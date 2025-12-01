package org.cloudbus.cloudsim.sdn.example;

import java.util.Calendar;
import java.util.LinkedList;
import java.util.List;

import org.cloudbus.cloudsim.DatacenterCharacteristics;
import org.cloudbus.cloudsim.Host;
import org.cloudbus.cloudsim.Storage;
import org.cloudbus.cloudsim.VmAllocationPolicy;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.sdn.CloudSimEx;
import org.cloudbus.cloudsim.sdn.HostFactory;
import org.cloudbus.cloudsim.sdn.HostFactorySimple;
import org.cloudbus.cloudsim.sdn.SDNBroker;
import org.cloudbus.cloudsim.sdn.monitor.power.PowerUtilizationMaxHostInterface;
import org.cloudbus.cloudsim.sdn.nos.NetworkOperatingSystem;
import org.cloudbus.cloudsim.sdn.nos.NetworkOperatingSystemSimple;
import org.cloudbus.cloudsim.sdn.parsers.PhysicalTopologyParser;
import org.cloudbus.cloudsim.sdn.physicalcomponents.SDNDatacenter;
import org.cloudbus.cloudsim.sdn.policies.selectlink.LinkSelectionPolicyCFRRL;
import org.cloudbus.cloudsim.sdn.policies.vmallocation.VmAllocationPolicyCombinedLeastFullFirst;
import org.cloudbus.cloudsim.sdn.rl.CFRRLIntervalEntity;
import org.cloudbus.cloudsim.sdn.rl.CFRRLLogger;
import org.cloudbus.cloudsim.sdn.rl.RLPipe;

/**
 * CFR-RL Example: Complete runner for CFR-RL with Abilene topology.
 * 
 * This example:
 * 1. Loads Abilene physical topology (12 nodes)
 * 2. Loads Abilene virtual topology with 132 pre-defined flows
 * 3. Uses custom parser to ensure flow IDs match model indices
 * 4. Starts Python RL agent
 * 5. Runs simulation with periodic RL updates
 * 
 * Files required:
 * - dataset-abilene/abilene-physical.json
 * - dataset-abilene/abilene-virtual.json
 * - dataset-abilene/abilene-workload.csv
 * - RL/agent.py
 * - RL/best_abilene_v2.pt
 */
public class CFRRLExample {
    
    // ==================== CONFIGURATION ====================
    private static final int K_CRITICAL = 8;          // Number of critical flows to select
    private static final double RL_INTERVAL = 5.0;    // Seconds between RL updates
    private static final String PYTHON_CMD = "python"; // Python command (or "python")
    private static final String PYTHON_SCRIPT = "RL/agent.py";
    
    // File paths
    protected static String physicalFile = "dataset-abilene/abilene-physical.json";
    protected static String virtualFile  = "dataset-abilene/abilene-virtual.json";
    protected static String[] workloadFiles = {
        "dataset-abilene/abilene-workload.csv"
    };
    
    // Simulation state
    protected static NetworkOperatingSystem nos;
    protected static PowerUtilizationMaxHostInterface maxHostHandler = null;
    protected static RLPipe pipe = null;
    
    public static void main(String[] args) {
        
        // ==================== STEP 1: Initialize Logging ====================
        CFRRLLogger.init("cfrrl_debug.log");
        CFRRLLogger.section("CFR-RL ABILENE SIMULATION STARTING");
        CloudSimEx.setStartTime();
        
        CFRRLLogger.info("Main", "Configuration:");
        CFRRLLogger.info("Main", "  K_CRITICAL = " + K_CRITICAL);
        CFRRLLogger.info("Main", "  RL_INTERVAL = " + RL_INTERVAL + " seconds");
        CFRRLLogger.info("Main", "  Physical topology: " + physicalFile);
        CFRRLLogger.info("Main", "  Virtual topology: " + virtualFile);
        CFRRLLogger.info("Main", "  Workload: " + workloadFiles[0]);
        
        try {
            // ==================== STEP 2: Initialize CloudSim ====================
            CFRRLLogger.section("INITIALIZING CLOUDSIM");
            
            int numUsers = 1;
            Calendar calendar = Calendar.getInstance();
            boolean traceFlag = false;
            
            CloudSim.init(numUsers, calendar, traceFlag);
            CFRRLLogger.info("Main", "CloudSim core initialized");
            
            // ==================== STEP 3: Create Network Operating System ====================
            CFRRLLogger.section("CREATING NETWORK COMPONENTS");
            
            HostFactory hostFactory = new HostFactorySimple();
            nos = new NetworkOperatingSystemSimple();
            
            // Load physical topology (Abilene: 12 nodes, 30 links)
            CFRRLLogger.info("Main", "Loading physical topology...");
            PhysicalTopologyParser.loadPhysicalTopologySingleDC(physicalFile, nos, hostFactory);
            CFRRLLogger.info("Main", "Physical topology loaded: " + nos.getHostList().size() + " hosts");
            
            // ==================== STEP 4: Create CFR-RL Policy ====================
            CFRRLLogger.section("CREATING CFR-RL COMPONENTS");
            
            LinkSelectionPolicyCFRRL cfrrlPolicy = new LinkSelectionPolicyCFRRL(K_CRITICAL);
            cfrrlPolicy.setVerboseLogging(true);
            nos.setLinkSelectionPolicy(cfrrlPolicy);
            CFRRLLogger.info("Main", "CFR-RL link selection policy created and attached to NOS");
            
            // ==================== STEP 5: Create VM Allocation Policy ====================
            VmAllocationPolicy vmPolicy = new VmAllocationPolicyCombinedLeastFullFirst(nos.getHostList());
            CFRRLLogger.info("Main", "VM allocation policy created");
            
            // ==================== STEP 6: Create Datacenter ====================
            SDNDatacenter datacenter = createDatacenter("Abilene_DC", nos, vmPolicy);
            CFRRLLogger.info("Main", "Datacenter created: " + datacenter.getName());
            
            // ==================== STEP 7: Create Broker ====================
            SDNBroker broker = new SDNBroker("Abilene_Broker");
            CFRRLLogger.info("Main", "Broker created: " + broker.getName());
            
            // ==================== STEP 8: Deploy Virtual Topology ====================
            CFRRLLogger.section("DEPLOYING VIRTUAL TOPOLOGY");
            CFRRLLogger.info("Main", "Virtual topology: " + virtualFile);
            CFRRLLogger.info("Main", "Flow IDs extracted from names (flow_X -> X) via modified parser");
            
            // Deploy via broker - the modified VirtualTopologyParser will extract flow IDs from names
            broker.submitDeployApplication(datacenter, virtualFile);
            CFRRLLogger.info("Main", "Virtual topology submitted to broker");
            
            // ==================== STEP 9: Submit Workload ====================
            CFRRLLogger.section("SUBMITTING WORKLOAD");
            for (String wf : workloadFiles) {
                CFRRLLogger.info("Main", "Submitting workload: " + wf);
                broker.submitRequests(wf);
            }
            
            // ==================== STEP 10: Start Python Agent ====================
            CFRRLLogger.section("STARTING PYTHON RL AGENT");
            CFRRLLogger.info("Main", "Command: " + PYTHON_CMD + " " + PYTHON_SCRIPT);
            
            pipe = new RLPipe(PYTHON_CMD);
            pipe.startPython();
            CFRRLLogger.info("Main", "Python agent process started");
            
            // Quick connection test
            testPythonConnection(pipe);
            
            // ==================== STEP 11: Create RL Interval Entity ====================
            CFRRLLogger.section("CREATING RL INTERVAL ENTITY");
            
            CFRRLIntervalEntity rlEntity = new CFRRLIntervalEntity(
                "CFR-RL-Entity",
                RL_INTERVAL,
                pipe,
                nos,
                cfrrlPolicy,
                K_CRITICAL
            );
            
            CFRRLLogger.info("Main", "RL entity created, adding to CloudSim...");
            // Note: CFRRLIntervalEntity extends SimEntity, so it auto-registers
            
            // ==================== STEP 12: Run Simulation ====================
            CFRRLLogger.section("STARTING SIMULATION");
            CFRRLLogger.info("Main", "CloudSim.startSimulation()...");
            
            double finishTime = CloudSim.startSimulation();
            CloudSim.stopSimulation();
            
            // ==================== STEP 13: Print Results ====================
            CFRRLLogger.section("SIMULATION COMPLETE");
            CFRRLLogger.info("Main", "Finish time: " + finishTime + " seconds");
            CFRRLLogger.info("Main", "Wall clock: " + CloudSimEx.getElapsedTimeString());
            
            // Print routing statistics
            CFRRLLogger.section("ROUTING STATISTICS");
            cfrrlPolicy.logStatistics();
            
            // Print summary
            CFRRLLogger.section("SUMMARY");
            CFRRLLogger.info("Main", "Simulation completed successfully");
            CFRRLLogger.info("Main", "K_CRITICAL: " + K_CRITICAL);
            CFRRLLogger.info("Main", "RL_INTERVAL: " + RL_INTERVAL + "s");
            CFRRLLogger.info("Main", String.format("Routing: %s", cfrrlPolicy.getStatistics()));
            
        } catch (Exception e) {
            CFRRLLogger.error("Main", "Error during simulation", e);
            e.printStackTrace();
            
        } finally {
            // ==================== CLEANUP ====================
            CFRRLLogger.section("CLEANUP");
            
            if (pipe != null) {
                try {
                    CFRRLLogger.info("Main", "Closing Python pipe...");
                    pipe.close();
                    CFRRLLogger.info("Main", "Python pipe closed");
                } catch (Exception e) {
                    CFRRLLogger.error("Main", "Error closing pipe", e);
                }
            }
            
            CFRRLLogger.section("CFR-RL SIMULATION FINISHED");
            CFRRLLogger.info("Main", "Log files:");
            CFRRLLogger.info("Main", "  Java:   " + CFRRLLogger.getLogFilePath());
            CFRRLLogger.info("Main", "  Python: cfrrl_agent.log");
            
            CFRRLLogger.close();
        }
    }
    
    /**
     * Create SDN Datacenter.
     */
    private static SDNDatacenter createDatacenter(
            String name,
            NetworkOperatingSystem nos,
            VmAllocationPolicy vmPolicy) {
        
        List<Host> hosts = nos.getHostList();
        CFRRLLogger.debug("Main", "Creating datacenter with " + hosts.size() + " hosts");
        
        String arch = "x86";
        String os = "Linux";
        String vmm = "Xen";
        double time_zone = 10.0;
        double cost = 3.0;
        double costPerMem = 0.05;
        double costPerStorage = 0.001;
        double costPerBw = 0.0;
        
        LinkedList<Storage> storageList = new LinkedList<Storage>();
        
        DatacenterCharacteristics characteristics = new DatacenterCharacteristics(
            arch, os, vmm, hosts,
            time_zone, cost, costPerMem, costPerStorage, costPerBw
        );
        
        SDNDatacenter datacenter = null;
        try {
            if (vmPolicy instanceof PowerUtilizationMaxHostInterface) {
                maxHostHandler = (PowerUtilizationMaxHostInterface) vmPolicy;
            }
            datacenter = new SDNDatacenter(name, characteristics, vmPolicy, storageList, 0, nos);
            nos.setDatacenter(datacenter);
        } catch (Exception e) {
            CFRRLLogger.error("Main", "Failed to create datacenter", e);
            e.printStackTrace();
        }
        
        return datacenter;
    }
    
    /**
     * Test Python connection with a minimal request.
     */
    private static void testPythonConnection(RLPipe pipe) {
        try {
            CFRRLLogger.info("Main", "Testing Python connection...");
            
            // Send minimal test state with 132 flows (all zeros except a few)
            StringBuilder testState = new StringBuilder();
            testState.append("{\"k\": ").append(K_CRITICAL);
            testState.append(", \"time\": 0.0");
            testState.append(", \"flows\": [");
            
            // Create 132 zero-demand flows (matching model expectations)
            for (int i = 0; i < 132; i++) {
                if (i > 0) testState.append(",");
                int src = i / 11;
                int dst = (i % 11) < src ? (i % 11) : (i % 11) + 1;
                testState.append("{\"id\":").append(i);
                testState.append(",\"src\":").append(src);
                testState.append(",\"dst\":").append(dst);
                testState.append(",\"bw\":").append(i < 5 ? 100000000 : 0);  // First 5 flows have demand
                testState.append(",\"features\":[0.1,0.5,0.6,1.0]}");
            }
            testState.append("]}");
            
            pipe.sendState(testState.toString());
            String response = pipe.receiveAction();
            
            if (response != null && response.contains("critical_flows")) {
                CFRRLLogger.info("Main", "Python connection test: SUCCESS");
                CFRRLLogger.debug("Main", "Response: " + response);
            } else {
                CFRRLLogger.warn("Main", "Python connection test: Unexpected response");
                CFRRLLogger.warn("Main", "Response: " + response);
            }
            
        } catch (Exception e) {
            CFRRLLogger.error("Main", "Python connection test: FAILED", e);
            throw new RuntimeException("Cannot connect to Python agent", e);
        }
    }
}