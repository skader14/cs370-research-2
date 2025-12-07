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
import org.cloudbus.cloudsim.sdn.rl.LatencyCollector;
import org.cloudbus.cloudsim.sdn.rl.RLPipe;

/**
 * CFR-RL Example with Latency Collection
 * 
 * This version adds LatencyCollector to measure per-packet queuing delays.
 * Use this to compare MLU-only vs latency-aware models.
 * 
 * CONFIGURATION:
 * 1. Set EXPERIMENT_NAME below to identify the run
 * 2. Edit RL/agent.py to select the model (lw0.0 or lw0.3)
 * 3. Run and compare latency_results_*.csv files
 */
public class CFRRLExampleLatency {
    
    // ==============================================================================
    // EXPERIMENT CONFIGURATION - EDIT THIS SECTION
    // ==============================================================================
    
    /**
     * Experiment name - used for output file naming.
     * Change this when switching between models!
     */
    // private static final String EXPERIMENT_NAME = "mlu_only";  // or "latency_aware"
    private static final String EXPERIMENT_NAME = "latency_aware";  // or "latency_aware"

    private static final String OUTPUT_DIR = "outputs";

    
    // Output files will be named:
    //   cfrrl_debug_{EXPERIMENT_NAME}.log
    //   latency_results_{EXPERIMENT_NAME}.csv
    //   cfrrl_agent_{EXPERIMENT_NAME}.log (set in agent.py)
    
    // ==============================================================================
    // STANDARD CONFIGURATION
    // ==============================================================================
    
    private static final int K_CRITICAL = 8;          // Number of critical flows to select
    private static final double RL_INTERVAL = 5.0;    // Seconds between RL updates
    private static final String PYTHON_CMD = "python"; // Python command
    private static final String PYTHON_SCRIPT = "RL/agent.py";
    
    // File paths
    protected static String physicalFile = "dataset-fattree/fattree-physical.json";
    protected static String virtualFile  = "dataset-fattree/fattree-virtual.json";
    protected static String[] workloadFiles = {
        "dataset-abilene/abilene-workload-heavy.csv"
    };
    
    // Simulation state
    protected static NetworkOperatingSystem nos;
    protected static PowerUtilizationMaxHostInterface maxHostHandler = null;
    protected static RLPipe pipe = null;

    /**
     * Ensure output directory exists
     */
    private static void ensureOutputDir() {
        java.io.File dir = new java.io.File(OUTPUT_DIR);
        if (!dir.exists()) {
            dir.mkdirs();
        }
    }
    
    public static void main(String[] args) {
        
        // Ensure output directory exists
        ensureOutputDir();

        // Derived file names from experiment name
        String logFile = OUTPUT_DIR + "/cfrrl_debug_" + EXPERIMENT_NAME + ".log";
        String latencyCsvFile = OUTPUT_DIR + "/latency_results_" + EXPERIMENT_NAME + ".csv";
        
        // ==================== STEP 1: Initialize Logging ====================
        CFRRLLogger.init(logFile);
        CFRRLLogger.section("CFR-RL LATENCY EXPERIMENT: " + EXPERIMENT_NAME);
        CloudSimEx.setStartTime();
        
        CFRRLLogger.info("Main", "==============================================");
        CFRRLLogger.info("Main", "EXPERIMENT: " + EXPERIMENT_NAME);
        CFRRLLogger.info("Main", "==============================================");
        CFRRLLogger.info("Main", "Configuration:");
        CFRRLLogger.info("Main", "  K_CRITICAL = " + K_CRITICAL);
        CFRRLLogger.info("Main", "  RL_INTERVAL = " + RL_INTERVAL + " seconds");
        CFRRLLogger.info("Main", "  Physical topology: " + physicalFile);
        CFRRLLogger.info("Main", "  Virtual topology: " + virtualFile);
        CFRRLLogger.info("Main", "  Workload: " + workloadFiles[0]);
        CFRRLLogger.info("Main", "Output files:");
        CFRRLLogger.info("Main", "  Log: " + logFile);
        CFRRLLogger.info("Main", "  Latency CSV: " + latencyCsvFile);
        
        // ==================== STEP 2: Initialize LatencyCollector ====================
        CFRRLLogger.section("INITIALIZING LATENCY COLLECTOR");
        
        LatencyCollector latencyCollector = LatencyCollector.getInstance();
        latencyCollector.initCsvExport(latencyCsvFile);
        CFRRLLogger.info("Main", "LatencyCollector initialized");
        CFRRLLogger.info("Main", "CSV export: " + latencyCsvFile);
        
        try {
            // ==================== STEP 3: Initialize CloudSim ====================
            CFRRLLogger.section("INITIALIZING CLOUDSIM");
            
            int numUsers = 1;
            Calendar calendar = Calendar.getInstance();
            boolean traceFlag = false;
            
            CloudSim.init(numUsers, calendar, traceFlag);
            CFRRLLogger.info("Main", "CloudSim core initialized");
            
            // ==================== STEP 4: Create Network Operating System ====================
            CFRRLLogger.section("CREATING NETWORK COMPONENTS");
            
            HostFactory hostFactory = new HostFactorySimple();
            nos = new NetworkOperatingSystemSimple();
            
            // Load physical topology (Abilene: 12 nodes, 30 links)
            CFRRLLogger.info("Main", "Loading physical topology...");
            PhysicalTopologyParser.loadPhysicalTopologySingleDC(physicalFile, nos, hostFactory);
            CFRRLLogger.info("Main", "Physical topology loaded: " + nos.getHostList().size() + " hosts");
            
            // ==================== STEP 5: Create CFR-RL Policy ====================
            CFRRLLogger.section("CREATING CFR-RL COMPONENTS");
            
            LinkSelectionPolicyCFRRL cfrrlPolicy = new LinkSelectionPolicyCFRRL(K_CRITICAL);
            cfrrlPolicy.setVerboseLogging(true);
            nos.setLinkSelectionPolicy(cfrrlPolicy);
            CFRRLLogger.info("Main", "CFR-RL link selection policy created and attached to NOS");
            
            // ==================== STEP 6: Create VM Allocation Policy ====================
            VmAllocationPolicy vmPolicy = new VmAllocationPolicyCombinedLeastFullFirst(nos.getHostList());
            CFRRLLogger.info("Main", "VM allocation policy created");
            
            // ==================== STEP 7: Create Datacenter ====================
            SDNDatacenter datacenter = createDatacenter("Abilene_DC", nos, vmPolicy);
            CFRRLLogger.info("Main", "Datacenter created: " + datacenter.getName());
            
            // Enable latency recording on datacenter
            datacenter.setLatencyRecordingEnabled(true);
            datacenter.setLatencyLogFrequency(50);  // Log every 50th packet
            CFRRLLogger.info("Main", "Latency recording enabled on datacenter");
            
            // ==================== STEP 8: Create Broker ====================
            SDNBroker broker = new SDNBroker("Abilene_Broker");
            CFRRLLogger.info("Main", "Broker created: " + broker.getName());
            
            // ==================== STEP 9: Deploy Virtual Topology ====================
            CFRRLLogger.section("DEPLOYING VIRTUAL TOPOLOGY");
            CFRRLLogger.info("Main", "Virtual topology: " + virtualFile);
            
            broker.submitDeployApplication(datacenter, virtualFile);
            CFRRLLogger.info("Main", "Virtual topology submitted to broker");
            
            // ==================== STEP 10: Submit Workload ====================
            CFRRLLogger.section("SUBMITTING WORKLOAD");
            for (String wf : workloadFiles) {
                CFRRLLogger.info("Main", "Submitting workload: " + wf);
                broker.submitRequests(wf);
            }
            
            // ==================== STEP 11: Start Python Agent ====================
            CFRRLLogger.section("STARTING PYTHON RL AGENT");
            CFRRLLogger.info("Main", "Command: " + PYTHON_CMD + " " + PYTHON_SCRIPT);
            CFRRLLogger.info("Main", "NOTE: Make sure agent.py is configured to use the correct model!");
            
            pipe = new RLPipe(PYTHON_CMD);
            pipe.startPython();
            CFRRLLogger.info("Main", "Python agent process started");
            
            // Quick connection test
            testPythonConnection(pipe);
            
            // ==================== STEP 12: Create RL Interval Entity ====================
            CFRRLLogger.section("CREATING RL INTERVAL ENTITY");
            
            CFRRLIntervalEntity rlEntity = new CFRRLIntervalEntity(
                "CFR-RL-Entity",
                RL_INTERVAL,
                pipe,
                nos,
                cfrrlPolicy,
                K_CRITICAL
            );
            
            CFRRLLogger.info("Main", "RL entity created");
            
            // ==================== STEP 13: Run Simulation ====================
            CFRRLLogger.section("STARTING SIMULATION");
            CFRRLLogger.info("Main", "CloudSim.startSimulation()...");
            
            double finishTime = CloudSim.startSimulation();
            CloudSim.stopSimulation();
            
            // ==================== STEP 14: Print Results ====================
            CFRRLLogger.section("SIMULATION COMPLETE");
            CFRRLLogger.info("Main", "Finish time: " + finishTime + " seconds");
            CFRRLLogger.info("Main", "Wall clock: " + CloudSimEx.getElapsedTimeString());
            
            // Print routing statistics
            CFRRLLogger.section("ROUTING STATISTICS");
            cfrrlPolicy.logStatistics();
            
            // ==================== STEP 15: Print Latency Statistics ====================
            CFRRLLogger.section("LATENCY STATISTICS");
            latencyCollector.printSummary();
            
            // Print summary
            CFRRLLogger.section("EXPERIMENT SUMMARY: " + EXPERIMENT_NAME);
            CFRRLLogger.info("Main", "Experiment: " + EXPERIMENT_NAME);
            CFRRLLogger.info("Main", "K_CRITICAL: " + K_CRITICAL);
            CFRRLLogger.info("Main", "RL_INTERVAL: " + RL_INTERVAL + "s");
            CFRRLLogger.info("Main", String.format("Routing: %s", cfrrlPolicy.getStatistics()));
            CFRRLLogger.info("Main", "");
            CFRRLLogger.info("Main", "Output files created:");
            CFRRLLogger.info("Main", "  " + logFile);
            CFRRLLogger.info("Main", "  " + latencyCsvFile);
            
        } catch (Exception e) {
            CFRRLLogger.error("Main", "Error during simulation", e);
            e.printStackTrace();
            
        } finally {
            // ==================== CLEANUP ====================
            CFRRLLogger.section("CLEANUP");
            
            // Close latency collector
            CFRRLLogger.info("Main", "Closing LatencyCollector...");
            latencyCollector.close();
            CFRRLLogger.info("Main", "LatencyCollector closed");
            
            if (pipe != null) {
                try {
                    CFRRLLogger.info("Main", "Closing Python pipe...");
                    pipe.close();
                    CFRRLLogger.info("Main", "Python pipe closed");
                } catch (Exception e) {
                    CFRRLLogger.error("Main", "Error closing pipe", e);
                }
            }
            
            CFRRLLogger.section("EXPERIMENT " + EXPERIMENT_NAME + " FINISHED");
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
            
            StringBuilder testState = new StringBuilder();
            testState.append("{\"k\": ").append(K_CRITICAL);
            testState.append(", \"time\": 0.0");
            testState.append(", \"flows\": [");
            
            for (int i = 0; i < 132; i++) {
                if (i > 0) testState.append(",");
                int src = i / 11;
                int dst = (i % 11) < src ? (i % 11) : (i % 11) + 1;
                testState.append("{\"id\":").append(i);
                testState.append(",\"src\":").append(src);
                testState.append(",\"dst\":").append(dst);
                testState.append(",\"bw\":").append(i < 5 ? 100000000 : 0);
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