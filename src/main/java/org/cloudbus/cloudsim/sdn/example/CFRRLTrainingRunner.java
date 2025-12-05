package org.cloudbus.cloudsim.sdn.example;

import java.io.File;
import java.io.FileReader;
import java.io.BufferedReader;
import java.io.IOException;
import java.util.Calendar;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import java.util.HashSet;

import org.cloudbus.cloudsim.DatacenterCharacteristics;
import org.cloudbus.cloudsim.Host;
import org.cloudbus.cloudsim.Storage;
import org.cloudbus.cloudsim.VmAllocationPolicy;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.sdn.CloudSimEx;
import org.cloudbus.cloudsim.sdn.HostFactory;
import org.cloudbus.cloudsim.sdn.HostFactorySimple;
import org.cloudbus.cloudsim.sdn.SDNBroker;
import org.cloudbus.cloudsim.sdn.nos.NetworkOperatingSystem;
import org.cloudbus.cloudsim.sdn.nos.NetworkOperatingSystemSimple;
import org.cloudbus.cloudsim.sdn.parsers.PhysicalTopologyParser;
import org.cloudbus.cloudsim.sdn.physicalcomponents.SDNDatacenter;
import org.cloudbus.cloudsim.sdn.policies.selectlink.LinkSelectionPolicyCFRRL;
import org.cloudbus.cloudsim.sdn.policies.vmallocation.VmAllocationPolicyCombinedLeastFullFirst;
import org.cloudbus.cloudsim.sdn.rl.CFRRLLogger;
import org.cloudbus.cloudsim.sdn.rl.LatencyCollector;
import org.cloudbus.cloudsim.sdn.rl.LinkStatsCollector;

/**
 * CFRRLTrainingRunner: Episode runner for CloudSim-in-the-loop RL training.
 * 
 * This runner is designed to be called from Python as a subprocess for each
 * training episode. It:
 * 
 * 1. Accepts command-line arguments:
 *    - workload file path (required)
 *    - critical flows file path (required)
 *    - output directory (optional, defaults to "outputs/")
 * 
 * 2. Runs a single simulation episode with the given workload and critical flows
 * 
 * 3. Exports statistics for Python to read:
 *    - episode_summary.json: Aggregate metrics for reward computation
 *    - flow_summary.csv: Per-flow statistics for feature computation
 *    - link_stats.csv: Per-link utilization for path features
 *    - latency_results.csv: Per-packet details (optional, for debugging)
 * 
 * 4. Exits with code 0 on success, non-zero on failure
 * 
 * Usage:
 *   java CFRRLTrainingRunner <workload.csv> <critical_flows.txt> [output_dir]
 * 
 * Example:
 *   java CFRRLTrainingRunner episode_42_workload.csv critical_flows.txt outputs/episode_42/
 * 
 * Critical flows file format (one flow ID per line):
 *   84
 *   50
 *   109
 *   17
 *   ...
 * 
 * @author CFR-RL Training Extension
 */
public class CFRRLTrainingRunner {
    
    // ==================== CONFIGURATION ====================
    private static final int K_CRITICAL = 8;           // Number of critical flows
    private static final double RL_INTERVAL = 5.0;     // Monitoring interval (not for RL updates in training)
    
    // Fixed topology files (Abilene)
    private static final String PHYSICAL_FILE = "dataset-abilene/abilene-physical.json";
    private static final String VIRTUAL_FILE = "dataset-abilene/abilene-virtual.json";
    
    // Simulation state
    private static NetworkOperatingSystem nos;
    private static LinkSelectionPolicyCFRRL cfrrlPolicy;
    private static String outputDir = "outputs/";
    
    public static void main(String[] args) {
        long startTime = System.currentTimeMillis();
        
        // ==================== PARSE ARGUMENTS ====================
        if (args.length < 2) {
            System.err.println("Usage: CFRRLTrainingRunner <workload.csv> <critical_flows.txt> [output_dir]");
            System.err.println("  workload.csv: Path to episode workload file");
            System.err.println("  critical_flows.txt: Path to file with critical flow IDs (one per line)");
            System.err.println("  output_dir: Directory for output files (default: outputs/)");
            System.exit(1);
        }
        
        String workloadFile = args[0];
        String criticalFlowsFile = args[1];
        if (args.length >= 3) {
            outputDir = args[2];
            if (!outputDir.endsWith("/")) {
                outputDir += "/";
            }
        }
        
        // Create output directory if needed
        new File(outputDir).mkdirs();
        
        // Initialize logging (minimal for training)
        String logFile = outputDir + "episode.log";
        CFRRLLogger.init(logFile);
        CFRRLLogger.info("TrainingRunner", "=== EPISODE START ===");
        CFRRLLogger.info("TrainingRunner", "Workload: " + workloadFile);
        CFRRLLogger.info("TrainingRunner", "Critical flows: " + criticalFlowsFile);
        CFRRLLogger.info("TrainingRunner", "Output dir: " + outputDir);
        
        try {
            // ==================== LOAD CRITICAL FLOWS ====================
            Set<Integer> criticalFlows = loadCriticalFlows(criticalFlowsFile);
            CFRRLLogger.info("TrainingRunner", "Loaded " + criticalFlows.size() + " critical flows: " + criticalFlows);
            
            // ==================== INITIALIZE COLLECTORS ====================
            LatencyCollector.reset();
            LinkStatsCollector.getInstance().reset();
            
            // Initialize CSV export for detailed packet data
            LatencyCollector.getInstance().initCsvExport(outputDir + "latency_results.csv");
            
            // Set critical flows for tracking
            LatencyCollector.getInstance().setCriticalFlows(criticalFlows);
            
            // ==================== INITIALIZE CLOUDSIM ====================
            CloudSim.init(1, Calendar.getInstance(), false);
            CloudSimEx.setStartTime();
            
            // ==================== CREATE NETWORK ====================
            HostFactory hostFactory = new HostFactorySimple();
            nos = new NetworkOperatingSystemSimple();
            
            // Load physical topology
            PhysicalTopologyParser.loadPhysicalTopologySingleDC(PHYSICAL_FILE, nos, hostFactory);
            CFRRLLogger.info("TrainingRunner", "Physical topology loaded: " + nos.getHostList().size() + " hosts");
            
            // ==================== CREATE CFR-RL POLICY ====================
            cfrrlPolicy = new LinkSelectionPolicyCFRRL(K_CRITICAL);
            cfrrlPolicy.setVerboseLogging(false);  // Quiet for training
            nos.setLinkSelectionPolicy(cfrrlPolicy);
            
            // Set critical flows from file (not from Python pipe)
            cfrrlPolicy.setCriticalFlows(criticalFlows);
            CFRRLLogger.info("TrainingRunner", "Critical flows set on policy");
            
            // ==================== CREATE DATACENTER ====================
            VmAllocationPolicy vmPolicy = new VmAllocationPolicyCombinedLeastFullFirst(nos.getHostList());
            SDNDatacenter datacenter = createDatacenter("TrainingDC", nos, vmPolicy);
            
            // ==================== CREATE BROKER ====================
            SDNBroker broker = new SDNBroker("TrainingBroker");
            
            // ==================== DEPLOY VIRTUAL TOPOLOGY ====================
            broker.submitDeployApplication(datacenter, VIRTUAL_FILE);
            CFRRLLogger.info("TrainingRunner", "Virtual topology deployed");
            
            // ==================== SUBMIT WORKLOAD ====================
            CFRRLLogger.info("TrainingRunner", "Submitting workload: " + workloadFile);
            broker.submitRequests(workloadFile);
            
            // ==================== RUN SIMULATION ====================
            CFRRLLogger.info("TrainingRunner", "Starting simulation...");
            double finishTime = CloudSim.startSimulation();
            CloudSim.stopSimulation();
            
            long wallClockMs = System.currentTimeMillis() - startTime;
            CFRRLLogger.info("TrainingRunner", String.format(
                "Simulation complete: %.2f sim seconds, %d ms wall clock",
                finishTime, wallClockMs));
            
            // ==================== EXPORT RESULTS ====================
            CFRRLLogger.info("TrainingRunner", "Exporting results...");
            
            // 1. Episode summary (for reward computation)
            LatencyCollector.getInstance().exportEpisodeSummary(outputDir + "episode_summary.json");
            
            // 2. Flow summary (for feature computation)
            LatencyCollector.getInstance().exportFlowSummary(outputDir + "flow_summary.csv");
            
            // 3. Link stats (for path utilization features)
            LinkStatsCollector.getInstance().exportToCsv(outputDir + "link_stats.csv");
            
            // 4. Print summaries to log
            LatencyCollector.getInstance().printSummary();
            LinkStatsCollector.getInstance().printSummary();
            
            // ==================== SUCCESS ====================
            CFRRLLogger.info("TrainingRunner", "=== EPISODE COMPLETE ===");
            CFRRLLogger.info("TrainingRunner", "Output files:");
            CFRRLLogger.info("TrainingRunner", "  " + outputDir + "episode_summary.json");
            CFRRLLogger.info("TrainingRunner", "  " + outputDir + "flow_summary.csv");
            CFRRLLogger.info("TrainingRunner", "  " + outputDir + "link_stats.csv");
            CFRRLLogger.info("TrainingRunner", "  " + outputDir + "latency_results.csv");
            
            CFRRLLogger.close();
            System.exit(0);
            
        } catch (Exception e) {
            CFRRLLogger.error("TrainingRunner", "Episode failed", e);
            e.printStackTrace();
            CFRRLLogger.close();
            System.exit(1);
        }
    }
    
    /**
     * Load critical flow IDs from file.
     * File format: one integer per line
     */
    private static Set<Integer> loadCriticalFlows(String filename) throws IOException {
        Set<Integer> flows = new HashSet<>();
        
        try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (!line.isEmpty() && !line.startsWith("#")) {
                    try {
                        flows.add(Integer.parseInt(line));
                    } catch (NumberFormatException e) {
                        CFRRLLogger.warn("TrainingRunner", "Invalid flow ID: " + line);
                    }
                }
            }
        }
        
        if (flows.size() != K_CRITICAL) {
            CFRRLLogger.warn("TrainingRunner", 
                "Expected " + K_CRITICAL + " critical flows, got " + flows.size());
        }
        
        return flows;
    }
    
    /**
     * Create SDN Datacenter.
     */
    private static SDNDatacenter createDatacenter(
            String name,
            NetworkOperatingSystem nos,
            VmAllocationPolicy vmPolicy) throws Exception {
        
        List<Host> hosts = nos.getHostList();
        
        DatacenterCharacteristics characteristics = new DatacenterCharacteristics(
            "x86", "Linux", "Xen", hosts,
            10.0, 3.0, 0.05, 0.001, 0.0
        );
        
        LinkedList<Storage> storageList = new LinkedList<Storage>();
        
        SDNDatacenter datacenter = new SDNDatacenter(
            name, characteristics, vmPolicy, storageList, 0, nos
        );
        nos.setDatacenter(datacenter);
        
        return datacenter;
    }
}