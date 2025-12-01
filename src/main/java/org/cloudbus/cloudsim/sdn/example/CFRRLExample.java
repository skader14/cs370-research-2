package org.cloudbus.cloudsim.sdn.example;

import java.util.Calendar;

import org.cloudbus.cloudsim.Log;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.sdn.nos.NetworkOperatingSystem;
import org.cloudbus.cloudsim.sdn.policies.selectlink.LinkSelectionPolicyCFRRL;
import org.cloudbus.cloudsim.sdn.rl.CFRRLIntervalEntity;
import org.cloudbus.cloudsim.sdn.rl.CFRRLLogger;
import org.cloudbus.cloudsim.sdn.rl.RLPipe;

/**
 * Example runner for CFR-RL integration with CloudSimSDN.
 * 
 * This demonstrates how to:
 * 1. Initialize CFR-RL logging
 * 2. Create the CFR-RL link selection policy
 * 3. Start the Python agent via RLPipe
 * 4. Create the interval entity for periodic updates
 * 5. Run the simulation
 * 
 * LOGGING:
 * - Java logs: cfrrl_debug.log
 * - Python logs: cfrrl_agent.log
 * - Console: Both Java and Python output
 */
public class CFRRLExample {
    
    // Configuration
    private static final int K_CRITICAL = 8;          // Number of critical flows
    private static final double RL_INTERVAL = 5.0;    // Seconds between RL updates
    private static final String PYTHON_SCRIPT = "RL/agent.py";

    protected static String phyiscalFile = "dataset-abilene/abilene-physical.json";
    protected static String virtualFile  = "dataset-abilene/abilene-virtual.json";

    protected static String[] workloadFiles = {
        "dataset-abilene/abilene-workload.csv"
    };
    
    public static void main(String[] args) {
        
        // ==================== STEP 1: Initialize Logging ====================
        CFRRLLogger.init("cfrrl_debug.log");
        CFRRLLogger.section("CFR-RL EXAMPLE STARTING");
        
        CFRRLLogger.info("Main", "Configuration:");
        CFRRLLogger.info("Main", "  K_CRITICAL = " + K_CRITICAL);
        CFRRLLogger.info("Main", "  RL_INTERVAL = " + RL_INTERVAL + " seconds");
        CFRRLLogger.info("Main", "  PYTHON_SCRIPT = " + PYTHON_SCRIPT);
        
        RLPipe pipe = null;
        
        try {
            // ==================== STEP 2: Initialize CloudSim ====================
            CFRRLLogger.section("INITIALIZING CLOUDSIM");
            
            int numUsers = 1;
            Calendar calendar = Calendar.getInstance();
            boolean traceFlag = false;
            
            CloudSim.init(numUsers, calendar, traceFlag);
            CFRRLLogger.info("Main", "CloudSim initialized");


            
            // ==================== STEP 3: Create Link Selection Policy ====================
            CFRRLLogger.section("CREATING CFR-RL COMPONENTS");
            
            LinkSelectionPolicyCFRRL cfrrlPolicy = new LinkSelectionPolicyCFRRL(K_CRITICAL);
            cfrrlPolicy.setVerboseLogging(true);  // Enable detailed logging
            CFRRLLogger.info("Main", "LinkSelectionPolicyCFRRL created");
            
            // ==================== STEP 4: Start Python Agent ====================
            CFRRLLogger.info("Main", "Starting Python agent...");
            
            pipe = new RLPipe("python");
            pipe.startPython();
            CFRRLLogger.info("Main", "Python agent started");
            
            // Test the connection with a simple ping
            CFRRLLogger.info("Main", "Testing Python connection...");
            testPythonConnection(pipe);
            
            // ==================== STEP 5: Create Network Operating System ====================
            // NOTE: You need to replace this with your actual NOS creation
            // This is just a placeholder showing where it would go
            
            CFRRLLogger.info("Main", "NOTE: You need to create your NOS and datacenter here");
            CFRRLLogger.info("Main", "Example:");
            CFRRLLogger.info("Main", "  NetworkOperatingSystem nos = new NetworkOperatingSystem(...);");
            CFRRLLogger.info("Main", "  nos.setLinkSelectionPolicy(cfrrlPolicy);");
            
            /*
            // Example of how to set up with your existing simulation:
            
            NetworkOperatingSystem nos = new SimpleNetworkOperatingSystem("NOS");
            nos.setLinkSelectionPolicy(cfrrlPolicy);  // Use our CFR-RL policy
            
            // Create the interval entity
            CFRRLIntervalEntity rlEntity = new CFRRLIntervalEntity(
                "CFR-RL-Entity",
                RL_INTERVAL,
                pipe,
                nos,
                cfrrlPolicy,
                K_CRITICAL
            );
            
            // Create your datacenter, VMs, workload, etc.
            // ...
            
            // Start simulation
            CFRRLLogger.section("STARTING SIMULATION");
            CloudSim.startSimulation();
            CloudSim.stopSimulation();
            
            // Print results
            CFRRLLogger.section("SIMULATION COMPLETE");
            cfrrlPolicy.logStatistics();
            */
            
            CFRRLLogger.info("Main", "");
            CFRRLLogger.info("Main", "To integrate with your simulation:");
            CFRRLLogger.info("Main", "1. Create your NOS with: nos.setLinkSelectionPolicy(cfrrlPolicy)");
            CFRRLLogger.info("Main", "2. Create CFRRLIntervalEntity with your NOS");
            CFRRLLogger.info("Main", "3. Run CloudSim.startSimulation()");
            
        } catch (Exception e) {
            CFRRLLogger.error("Main", "Error during simulation", e);
            
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
            
            CFRRLLogger.section("CFR-RL EXAMPLE FINISHED");
            CFRRLLogger.info("Main", "");
            CFRRLLogger.info("Main", "Log files created:");
            CFRRLLogger.info("Main", "  Java:   " + CFRRLLogger.getLogFilePath());
            CFRRLLogger.info("Main", "  Python: cfrrl_agent.log (in working directory)");
            
            CFRRLLogger.close();
        }
    }
    
    /**
     * Test the Python connection with a simple request.
     */
    private static void testPythonConnection(RLPipe pipe) {
        try {
            // Send a minimal test state
            String testState = "{\"k\": 2, \"flows\": [{\"id\": 1, \"bw\": 1000000}], \"time\": 0.0}";
            CFRRLLogger.debug("Main", "Sending test state: " + testState);
            
            pipe.sendState(testState);
            String response = pipe.receiveAction();
            
            CFRRLLogger.debug("Main", "Received response: " + response);
            
            if (response != null && response.contains("critical_flows")) {
                CFRRLLogger.info("Main", "Python connection test: SUCCESS");
            } else {
                CFRRLLogger.warn("Main", "Python connection test: Unexpected response");
            }
            
        } catch (Exception e) {
            CFRRLLogger.error("Main", "Python connection test: FAILED", e);
        }
    }
}