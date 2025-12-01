package org.cloudbus.cloudsim.sdn.rl;

import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.core.SimEntity;
import org.cloudbus.cloudsim.core.SimEvent;
import org.cloudbus.cloudsim.sdn.CloudSimEx;
import org.cloudbus.cloudsim.sdn.nos.ChannelManager;
import org.cloudbus.cloudsim.sdn.nos.NetworkOperatingSystem;
import org.cloudbus.cloudsim.sdn.physicalcomponents.Link;
import org.cloudbus.cloudsim.sdn.physicalcomponents.Node;
import org.cloudbus.cloudsim.sdn.physicalcomponents.SDNHost;
import org.cloudbus.cloudsim.sdn.policies.selectlink.LinkSelectionPolicyCFRRL;
import org.cloudbus.cloudsim.sdn.virtualcomponents.Channel;

import java.util.*;

/**
 * CFR-RL Interval Entity - Periodically updates critical flow selection.
 * 
 * IMPORTANT: This version always emits exactly 132 flow slots to match
 * the Abilene topology model expectations (12 nodes * 11 destinations = 132 flows).
 * Inactive flows are represented with zero bandwidth and default features.
 * 
 * Flow Index Mapping:
 *   flowIdx = src * 11 + (dst if dst < src else dst - 1)
 *   
 * Node Index to Name:
 *   0: Seattle, 1: Sunnyvale, 2: LosAngeles, 3: Denver,
 *   4: KansasCity, 5: Houston, 6: Chicago, 7: Indianapolis,
 *   8: Atlanta, 9: Washington, 10: NewYork, 11: Jacksonville
 * 
 * TERMINATION: Stops when max simulation time reached OR after consecutive idle intervals.
 */
public class CFRRLIntervalEntity extends SimEntity {

    private static final String LOG_TAG = "IntervalEntity";
    private static final int CFRRL_UPDATE_EVENT = 20002;
    
    // Abilene topology constants
    private static final int NUM_NODES = 12;
    private static final int NUM_FLOWS = 132;  // 12 * 11
    
    // Termination constants
    private static final double DEFAULT_MAX_SIM_TIME = 120.0;  // 2 minutes default
    private static final int MAX_CONSECUTIVE_IDLE = 3;  // Stop after 3 idle intervals
    
    private double interval;
    private RLPipe pipe;
    private NetworkOperatingSystem nos;
    private LinkSelectionPolicyCFRRL cfrrlPolicy;
    private int kCritical;
    private int updateCounter = 0;
    
    // Termination tracking
    private double maxSimulationTime;
    private int consecutiveIdleCount = 0;
    
    // Normalization constants (matching training)
    private double maxDemandSeen = 1e9;      // 1 Gbps default
    private double maxCapacitySeen = 10e9;   // 10 Gbps default
    
    // VM ID to Node Index mapping (populated during first update)
    private Map<Integer, Integer> vmIdToNodeIndex = new HashMap<>();
    private boolean mappingInitialized = false;

    public CFRRLIntervalEntity(String name, double interval, RLPipe pipe, 
                                NetworkOperatingSystem nos,
                                LinkSelectionPolicyCFRRL cfrrlPolicy,
                                int kCritical) {
        this(name, interval, pipe, nos, cfrrlPolicy, kCritical, DEFAULT_MAX_SIM_TIME);
    }
    
    public CFRRLIntervalEntity(String name, double interval, RLPipe pipe, 
                                NetworkOperatingSystem nos,
                                LinkSelectionPolicyCFRRL cfrrlPolicy,
                                int kCritical, double maxSimulationTime) {
        super(name);
        this.interval = interval;
        this.pipe = pipe;
        this.nos = nos;
        this.cfrrlPolicy = cfrrlPolicy;
        this.kCritical = kCritical;
        this.maxSimulationTime = maxSimulationTime;
        
        CFRRLLogger.info(LOG_TAG, "Created with interval=" + interval + "s, K=" + kCritical);
        CFRRLLogger.info(LOG_TAG, "Max simulation time: " + maxSimulationTime + "s");
        CFRRLLogger.info(LOG_TAG, "Will always emit " + NUM_FLOWS + " flow slots");
    }

    @Override
    public void startEntity() {
        CFRRLLogger.section("CFR-RL INTERVAL ENTITY STARTED");
        CFRRLLogger.info(LOG_TAG, "Simulation time: " + CloudSim.clock());
        CFRRLLogger.info(LOG_TAG, "Update interval: " + interval + " seconds");
        CFRRLLogger.info(LOG_TAG, "Critical flows K: " + kCritical);
        CFRRLLogger.info(LOG_TAG, "Scheduling first update at t=" + (CloudSim.clock() + interval));
        
        schedule(getId(), interval, CFRRL_UPDATE_EVENT);
    }

    @Override
    public void processEvent(SimEvent ev) {
        if (ev.getTag() == CFRRL_UPDATE_EVENT) {
            CFRRLLogger.debug(LOG_TAG, "--- Processing RL Update Event ---");
            
            double currentTime = CloudSim.clock();
            
            // === TERMINATION CHECK 1: Maximum simulation time ===
            if (currentTime >= maxSimulationTime) {
                CFRRLLogger.info(LOG_TAG, "=== STOPPING: Max simulation time reached (" + 
                               currentTime + " >= " + maxSimulationTime + ") ===");
                printFinalSummary();
                return;  // Don't schedule next update
            }
            
            performRLUpdate();
            
            // Check activity for termination
            int activeChannels = (int) nos.getChannelManager().getTotalChannelNum();
            
            CFRRLLogger.debug(LOG_TAG, "Active channels: " + activeChannels);
            
            // === TERMINATION CHECK 2: Consecutive idle detection ===
            if (activeChannels == 0) {
                consecutiveIdleCount++;
                CFRRLLogger.debug(LOG_TAG, "Idle interval #" + consecutiveIdleCount + 
                                " of " + MAX_CONSECUTIVE_IDLE);
                
                if (consecutiveIdleCount >= MAX_CONSECUTIVE_IDLE) {
                    CFRRLLogger.info(LOG_TAG, "=== STOPPING: " + MAX_CONSECUTIVE_IDLE + 
                                   " consecutive idle intervals ===");
                    printFinalSummary();
                    return;  // Don't schedule next update
                }
            } else {
                // Reset idle counter when there's activity
                if (consecutiveIdleCount > 0) {
                    CFRRLLogger.debug(LOG_TAG, "Activity detected, resetting idle counter");
                }
                consecutiveIdleCount = 0;
            }
            
            // Schedule next update
            double nextTime = currentTime + interval;
            CFRRLLogger.debug(LOG_TAG, "Scheduling next update at t=" + nextTime);
            schedule(getId(), interval, CFRRL_UPDATE_EVENT);
        }
    }
    
    private void printFinalSummary() {
        CFRRLLogger.section("SIMULATION COMPLETE");
        CFRRLLogger.info(LOG_TAG, "Total RL updates: " + updateCounter);
        CFRRLLogger.info(LOG_TAG, "Final simulation time: " + CloudSim.clock());
        if (cfrrlPolicy != null) {
            CFRRLLogger.info(LOG_TAG, "Final routing stats: " + cfrrlPolicy.getStatistics());
        }
    }

    /**
     * Initialize VM ID to Node Index mapping.
     * Assumes VMs are named vm_0, vm_1, ..., vm_11 corresponding to nodes 0-11.
     */
    private void initializeVmMapping() {
        if (mappingInitialized) return;
        
        Map<String, Integer> vmNameToId = NetworkOperatingSystem.getVmNameToIdMap();
        CFRRLLogger.debug(LOG_TAG, "Initializing VM mapping from " + vmNameToId.size() + " VMs");
        
        for (Map.Entry<String, Integer> entry : vmNameToId.entrySet()) {
            String vmName = entry.getKey();
            Integer vmId = entry.getValue();
            
            // Parse node index from VM name (e.g., "vm_5" -> 5)
            if (vmName.startsWith("vm_")) {
                try {
                    int nodeIndex = Integer.parseInt(vmName.substring(3));
                    if (nodeIndex >= 0 && nodeIndex < NUM_NODES) {
                        vmIdToNodeIndex.put(vmId, nodeIndex);
                        CFRRLLogger.debug(LOG_TAG, "Mapped VM " + vmName + " (id=" + vmId + ") to node " + nodeIndex);
                    }
                } catch (NumberFormatException e) {
                    CFRRLLogger.warn(LOG_TAG, "Could not parse node index from VM name: " + vmName);
                }
            }
        }
        
        mappingInitialized = true;
        CFRRLLogger.info(LOG_TAG, "VM mapping initialized: " + vmIdToNodeIndex.size() + " VMs mapped");
    }

    /**
     * Compute flow index from source and destination node indices.
     * Matches the Python model's flow indexing: flowIdx = src * 11 + (dst if dst < src else dst - 1)
     */
    private int computeFlowIndex(int srcNode, int dstNode) {
        if (srcNode == dstNode) {
            throw new IllegalArgumentException("src and dst cannot be the same");
        }
        if (dstNode < srcNode) {
            return srcNode * 11 + dstNode;
        } else {
            return srcNode * 11 + (dstNode - 1);
        }
    }

    /**
     * Get source and destination node indices from flow index.
     */
    private int[] getNodePairFromFlowIndex(int flowIdx) {
        int srcNode = flowIdx / 11;
        int offset = flowIdx % 11;
        int dstNode;
        if (offset < srcNode) {
            dstNode = offset;
        } else {
            dstNode = offset + 1;
        }
        return new int[]{srcNode, dstNode};
    }

    /**
     * Get node index from VM ID.
     */
    private int getNodeIndexFromVmId(int vmId) {
        Integer nodeIdx = vmIdToNodeIndex.get(vmId);
        if (nodeIdx == null) {
            CFRRLLogger.warn(LOG_TAG, "Unknown VM ID: " + vmId + ", defaulting to -1");
            return -1;
        }
        return nodeIdx;
    }

    private void performRLUpdate() {
        updateCounter++;
        double currentTime = CloudSim.clock();
        
        CFRRLLogger.section("RL UPDATE #" + updateCounter);
        CFRRLLogger.info(LOG_TAG, "Simulation time: " + currentTime);
        
        // Initialize mapping on first update
        initializeVmMapping();
        
        try {
            // Step 1: Gather state (always 132 flows)
            CFRRLLogger.debug(LOG_TAG, "Step 1: Gathering network state (all " + NUM_FLOWS + " flows)...");
            String stateJson = buildStateJson();
            CFRRLLogger.logJson(LOG_TAG, "State JSON", stateJson);
            
            // Step 2: Send to Python
            CFRRLLogger.debug(LOG_TAG, "Step 2: Sending to Python...");
            long sendStart = System.currentTimeMillis();
            pipe.sendState(stateJson);
            CFRRLLogger.debug(LOG_TAG, "Sent in " + (System.currentTimeMillis() - sendStart) + "ms");
            
            // Step 3: Receive response
            CFRRLLogger.debug(LOG_TAG, "Step 3: Waiting for Python response...");
            long recvStart = System.currentTimeMillis();
            String response = pipe.receiveAction();
            CFRRLLogger.debug(LOG_TAG, "Received in " + (System.currentTimeMillis() - recvStart) + "ms");
            CFRRLLogger.logJson(LOG_TAG, "Response JSON", response);
            
            // Step 4: Parse response
            CFRRLLogger.debug(LOG_TAG, "Step 4: Parsing response...");
            Set<Integer> criticalFlows = parseResponse(response);
            CFRRLLogger.info(LOG_TAG, "Critical flows selected: " + criticalFlows);
            
            // Step 5: Update policy
            CFRRLLogger.debug(LOG_TAG, "Step 5: Updating policy...");
            if (cfrrlPolicy != null) {
                cfrrlPolicy.setCriticalFlows(criticalFlows);
                CFRRLLogger.info(LOG_TAG, "Policy updated with " + criticalFlows.size() + " critical flows");
            } else {
                CFRRLLogger.warn(LOG_TAG, "cfrrlPolicy is NULL - cannot update!");
            }
            
            // Summary
            int activeChannels = (int) nos.getChannelManager().getTotalChannelNum();
            CFRRLLogger.info(LOG_TAG, "Update #" + updateCounter + " complete: " + 
                           criticalFlows.size() + "/" + kCritical + " critical flows, " +
                           activeChannels + " active channels");
            
        } catch (Exception e) {
            CFRRLLogger.error(LOG_TAG, "Error during update #" + updateCounter, e);
        }
    }

    /**
     * Build state JSON with exactly 132 flow slots.
     * Active flows get real data, inactive flows get zeros.
     */
    private String buildStateJson() {
        StringBuilder sb = new StringBuilder();
        sb.append("{");
        
        sb.append("\"time\":").append(CloudSim.clock()).append(",");
        sb.append("\"update_num\":").append(updateCounter).append(",");
        sb.append("\"k\":").append(kCritical).append(",");
        
        // Initialize all 132 flow slots with zeros
        FlowFeatures[] allFlows = new FlowFeatures[NUM_FLOWS];
        for (int flowIdx = 0; flowIdx < NUM_FLOWS; flowIdx++) {
            int[] nodePair = getNodePairFromFlowIndex(flowIdx);
            allFlows[flowIdx] = createEmptyFlowFeatures(flowIdx, nodePair[0], nodePair[1]);
        }
        
        // Populate active flows from channels
        int activeCount = populateActiveFlows(allFlows);
        CFRRLLogger.debug(LOG_TAG, "Active flows: " + activeCount + " / " + NUM_FLOWS);
        
        // Log active flow details
        if (activeCount > 0) {
            CFRRLLogger.tableHeader(LOG_TAG, "FlowID", "Src", "Dst", "Demand", "PathLen");
            for (FlowFeatures ff : allFlows) {
                if (ff.demand > 0) {
                    CFRRLLogger.tableRow(LOG_TAG, ff.flowId, ff.srcNode, ff.dstNode,
                                        String.format("%.0f", ff.demand), ff.pathLength);
                }
            }
        }
        
        // Serialize all 132 flows
        sb.append("\"flows\":[");
        for (int i = 0; i < NUM_FLOWS; i++) {
            if (i > 0) sb.append(",");
            FlowFeatures ff = allFlows[i];
            
            sb.append("{");
            sb.append("\"id\":").append(ff.flowId).append(",");
            sb.append("\"src\":").append(ff.srcNode).append(",");
            sb.append("\"dst\":").append(ff.dstNode).append(",");
            sb.append("\"bw\":").append((long) ff.demand).append(",");
            sb.append("\"path_len\":").append(ff.pathLength).append(",");
            sb.append("\"bottleneck\":").append((long) ff.bottleneckCap).append(",");
            sb.append("\"num_paths\":").append(ff.numPaths).append(",");
            sb.append("\"features\":[");
            sb.append(String.format("%.6f", Math.min(ff.demand / maxDemandSeen, 1.0))).append(",");
            sb.append(String.format("%.6f", Math.min(ff.numPaths / 4.0, 1.0))).append(",");
            sb.append(String.format("%.6f", Math.min(ff.pathLength / 5.0, 1.0))).append(",");
            sb.append(String.format("%.6f", Math.min(ff.bottleneckCap / maxCapacitySeen, 1.0)));
            sb.append("]}");
        }
        sb.append("],");
        
        // Add link utilization info
        sb.append("\"links\":[");
        Collection<Link> links = nos.getPhysicalTopology().getAllLinks();
        int linkCount = 0;
        double maxUtil = 0;
        for (Link link : links) {
            if (linkCount > 0) sb.append(",");
            linkCount++;
            
            Node lowNode = link.getLowOrder();
            Node highNode = link.getHighOrder();
            
            double upBw = link.getBw(lowNode);
            double upFreeBw = link.getFreeBandwidth(lowNode);
            double upUtil = upBw > 0 ? (upBw - upFreeBw) / upBw : 0;
            
            double downBw = link.getBw(highNode);
            double downFreeBw = link.getFreeBandwidth(highNode);
            double downUtil = downBw > 0 ? (downBw - downFreeBw) / downBw : 0;
            
            double linkMaxUtil = Math.max(upUtil, downUtil);
            if (linkMaxUtil > maxUtil) maxUtil = linkMaxUtil;
            
            sb.append("{");
            sb.append("\"up_util\":").append(String.format("%.4f", upUtil)).append(",");
            sb.append("\"down_util\":").append(String.format("%.4f", downUtil));
            sb.append("}");
        }
        sb.append("],");
        
        CFRRLLogger.debug(LOG_TAG, "Links: " + linkCount + ", Max utilization: " + 
                         String.format("%.2f%%", maxUtil * 100));
        
        sb.append("\"active_flows\":").append(activeCount).append(",");
        sb.append("\"total_flows\":").append(NUM_FLOWS).append(",");
        sb.append("\"total_channels\":").append(nos.getChannelManager().getTotalChannelNum());
        sb.append("}");
        
        return sb.toString();
    }

    /**
     * Create empty flow features for an inactive flow.
     */
    private FlowFeatures createEmptyFlowFeatures(int flowIdx, int srcNode, int dstNode) {
        FlowFeatures ff = new FlowFeatures();
        ff.flowId = flowIdx;
        ff.srcNode = srcNode;
        ff.dstNode = dstNode;
        ff.demand = 0;
        ff.pathLength = 0;
        ff.bottleneckCap = 0;
        ff.numPaths = 0;
        return ff;
    }

    /**
     * Populate active flows from current channels.
     * Returns count of active flows.
     */
    private int populateActiveFlows(FlowFeatures[] allFlows) {
        ChannelManager cm = nos.getChannelManager();
        Map<String, Integer> vmNameToId = NetworkOperatingSystem.getVmNameToIdMap();
        Set<Integer> processedFlows = new HashSet<>();
        int activeCount = 0;
        
        for (Integer vmId : vmNameToId.values()) {
            if (vmId == null || vmId == -1) continue;
            
            List<Channel> channels = cm.findAllChannels(vmId);
            
            for (Channel ch : channels) {
                int chFlowId = ch.getChId();
                if (chFlowId == -1) continue;  // Skip default channels
                if (processedFlows.contains(chFlowId)) continue;
                processedFlows.add(chFlowId);
                
                // Get node indices from VM IDs
                int srcNode = getNodeIndexFromVmId(ch.getSrcId());
                int dstNode = getNodeIndexFromVmId(ch.getDstId());
                
                if (srcNode < 0 || dstNode < 0 || srcNode == dstNode) {
                    CFRRLLogger.warn(LOG_TAG, "Invalid node pair for channel: src=" + srcNode + ", dst=" + dstNode);
                    continue;
                }
                
                // Compute flow index
                int flowIdx = computeFlowIndex(srcNode, dstNode);
                
                // Verify the flow ID matches what we expect
                // Note: CloudSimSDN flowId from workload may differ from our computed index
                // We use the computed index based on src/dst nodes
                if (flowIdx < 0 || flowIdx >= NUM_FLOWS) {
                    CFRRLLogger.warn(LOG_TAG, "Flow index out of range: " + flowIdx);
                    continue;
                }
                
                // Update flow features
                FlowFeatures ff = allFlows[flowIdx];
                ff.demand = ch.getRequestedBandwidth();
                ff.pathLength = getPathLength(ch);
                ff.bottleneckCap = getBottleneckCapacity(ch);
                ff.numPaths = estimateNumPaths(srcNode, dstNode);
                
                // Update normalization constants
                if (ff.demand > maxDemandSeen) {
                    maxDemandSeen = ff.demand;
                }
                if (ff.bottleneckCap > maxCapacitySeen) {
                    maxCapacitySeen = ff.bottleneckCap;
                }
                
                activeCount++;
            }
        }
        
        return activeCount;
    }
    
    private int getPathLength(Channel ch) {
        try {
            return ch.getPathLength();
        } catch (NoSuchMethodError e) {
            CFRRLLogger.debug(LOG_TAG, "Channel.getPathLength() not available, using default");
            return 3;
        }
    }
    
    private double getBottleneckCapacity(Channel ch) {
        try {
            return ch.getBottleneckCapacity();
        } catch (NoSuchMethodError e) {
            CFRRLLogger.debug(LOG_TAG, "Channel.getBottleneckCapacity() not available, using default");
            return Math.max(ch.getAllocatedBandwidth() * 10, 10e9);
        }
    }
    
    private int estimateNumPaths(int srcNode, int dstNode) {
        try {
            // Try to get actual path count from topology
            // For now, use heuristic based on Abilene topology
            // Most node pairs have 2-4 paths
            return 2;  // Conservative default
        } catch (Exception e) {
            return 2;
        }
    }

    private Set<Integer> parseResponse(String response) {
        Set<Integer> criticalFlows = new HashSet<>();
        
        try {
            int start = response.indexOf("\"critical_flows\"");
            if (start < 0) {
                CFRRLLogger.warn(LOG_TAG, "No 'critical_flows' key in response!");
                return criticalFlows;
            }
            
            int arrayStart = response.indexOf("[", start);
            int arrayEnd = response.indexOf("]", arrayStart);
            
            if (arrayStart < 0 || arrayEnd < 0) {
                CFRRLLogger.warn(LOG_TAG, "Could not find array brackets in response");
                return criticalFlows;
            }
            
            String arrayContent = response.substring(arrayStart + 1, arrayEnd).trim();
            CFRRLLogger.debug(LOG_TAG, "Array content: " + arrayContent);
            
            if (arrayContent.isEmpty()) {
                CFRRLLogger.debug(LOG_TAG, "Empty array - no critical flows");
                return criticalFlows;
            }
            
            String[] items = arrayContent.split(",");
            
            for (String item : items) {
                item = item.trim().replace("\"", "");
                if (item.isEmpty()) continue;
                
                try {
                    int flowId = Integer.parseInt(item);
                    if (flowId >= 0 && flowId < NUM_FLOWS) {
                        criticalFlows.add(flowId);
                        CFRRLLogger.debug(LOG_TAG, "Parsed flow ID: " + flowId);
                    } else {
                        CFRRLLogger.warn(LOG_TAG, "Flow ID out of range: " + flowId);
                    }
                } catch (NumberFormatException e) {
                    CFRRLLogger.warn(LOG_TAG, "Could not parse flow ID: " + item);
                }
            }
            
        } catch (Exception e) {
            CFRRLLogger.error(LOG_TAG, "Error parsing response", e);
        }
        
        return criticalFlows;
    }

    @Override
    public void shutdownEntity() {
        CFRRLLogger.section("CFR-RL INTERVAL ENTITY SHUTDOWN");
        CFRRLLogger.info(LOG_TAG, "Total updates performed: " + updateCounter);
        if (cfrrlPolicy != null) {
            CFRRLLogger.info(LOG_TAG, "Final routing stats: " + cfrrlPolicy.getStatistics());
        }
    }

    /**
     * Flow features data class.
     */
    private static class FlowFeatures {
        int flowId;
        int srcNode;
        int dstNode;
        double demand;
        int pathLength;
        double bottleneckCap;
        int numPaths;
    }
}