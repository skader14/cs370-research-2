package org.cloudbus.cloudsim.sdn.rl;

import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.core.SimEntity;
import org.cloudbus.cloudsim.core.SimEvent;
import org.cloudbus.cloudsim.sdn.CloudSimEx;
import org.cloudbus.cloudsim.sdn.nos.ChannelManager;
import org.cloudbus.cloudsim.sdn.nos.NetworkOperatingSystem;
import org.cloudbus.cloudsim.sdn.physicalcomponents.Link;
import org.cloudbus.cloudsim.sdn.physicalcomponents.Node;
import org.cloudbus.cloudsim.sdn.physicalcomponents.PhysicalTopology;
import org.cloudbus.cloudsim.sdn.physicalcomponents.SDNHost;
import org.cloudbus.cloudsim.sdn.policies.selectlink.LinkSelectionPolicyCFRRL;
import org.cloudbus.cloudsim.sdn.virtualcomponents.Channel;

import java.util.*;

/**
 * CFR-RL Interval Entity - Periodically updates critical flow selection.
 * 
 * With comprehensive logging for debugging.
 */
public class CFRRLIntervalEntity extends SimEntity {

    private static final String LOG_TAG = "IntervalEntity";
    private static final int CFRRL_UPDATE_EVENT = 20002;
    
    private double interval;
    private RLPipe pipe;
    private NetworkOperatingSystem nos;
    private LinkSelectionPolicyCFRRL cfrrlPolicy;
    private int kCritical;
    private int updateCounter = 0;
    
    private double maxDemandSeen = 1e9;
    private double maxCapacitySeen = 10e9;

    public CFRRLIntervalEntity(String name, double interval, RLPipe pipe, 
                                NetworkOperatingSystem nos,
                                LinkSelectionPolicyCFRRL cfrrlPolicy,
                                int kCritical) {
        super(name);
        this.interval = interval;
        this.pipe = pipe;
        this.nos = nos;
        this.cfrrlPolicy = cfrrlPolicy;
        this.kCritical = kCritical;
        
        CFRRLLogger.info(LOG_TAG, "Created with interval=" + interval + "s, K=" + kCritical);
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
            
            performRLUpdate();
            
            // Check if we should continue
            boolean hasMoreEvents = CloudSimEx.hasMoreEvent(CFRRL_UPDATE_EVENT);
            int activeChannels = (int) nos.getChannelManager().getTotalChannelNum();
            
            CFRRLLogger.debug(LOG_TAG, "Has more events (excluding RL): " + hasMoreEvents);
            CFRRLLogger.debug(LOG_TAG, "Active channels: " + activeChannels);
            
            if (hasMoreEvents && activeChannels > 0) {
                double nextTime = CloudSim.clock() + interval;
                CFRRLLogger.debug(LOG_TAG, "Scheduling next update at t=" + nextTime);
                schedule(getId(), interval, CFRRL_UPDATE_EVENT);
            } else {
                CFRRLLogger.info(LOG_TAG, "STOPPING: hasMoreEvents=" + hasMoreEvents + 
                               ", activeChannels=" + activeChannels);
            }
        }
    }

    private void performRLUpdate() {
        updateCounter++;
        double currentTime = CloudSim.clock();
        
        CFRRLLogger.section("RL UPDATE #" + updateCounter);
        CFRRLLogger.info(LOG_TAG, "Simulation time: " + currentTime);
        
        try {
            // Step 1: Gather state
            CFRRLLogger.debug(LOG_TAG, "Step 1: Gathering network state...");
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

    private String buildStateJson() {
        StringBuilder sb = new StringBuilder();
        sb.append("{");
        
        sb.append("\"time\":").append(CloudSim.clock()).append(",");
        sb.append("\"update_num\":").append(updateCounter).append(",");
        sb.append("\"k\":").append(kCritical).append(",");
        
        // Extract flow features
        List<FlowFeatures> flowFeaturesList = extractFlowFeatures();
        CFRRLLogger.debug(LOG_TAG, "Extracted " + flowFeaturesList.size() + " flows with features");
        
        // Log flow details
        if (!flowFeaturesList.isEmpty()) {
            CFRRLLogger.tableHeader(LOG_TAG, "FlowID", "Demand", "PathLen", "Bottleneck", "NumPaths");
            for (FlowFeatures ff : flowFeaturesList) {
                CFRRLLogger.tableRow(LOG_TAG, ff.flowId, 
                                    String.format("%.0f", ff.demand),
                                    ff.pathLength, 
                                    String.format("%.0f", ff.bottleneckCap),
                                    ff.numPaths);
            }
        }
        
        // Flows JSON
        sb.append("\"flows\":[");
        boolean first = true;
        for (FlowFeatures ff : flowFeaturesList) {
            if (!first) sb.append(",");
            first = false;
            
            sb.append("{");
            sb.append("\"id\":").append(ff.flowId).append(",");
            sb.append("\"src\":").append(ff.srcVm).append(",");
            sb.append("\"dst\":").append(ff.dstVm).append(",");
            sb.append("\"bw\":").append((long) ff.demand).append(",");
            sb.append("\"path_len\":").append(ff.pathLength).append(",");
            sb.append("\"bottleneck\":").append((long) ff.bottleneckCap).append(",");
            sb.append("\"num_paths\":").append(ff.numPaths).append(",");
            sb.append("\"features\":[");
            sb.append(String.format("%.4f", ff.demand / maxDemandSeen)).append(",");
            sb.append(String.format("%.4f", ff.numPaths / 4.0)).append(",");
            sb.append(String.format("%.4f", ff.pathLength / 5.0)).append(",");
            sb.append(String.format("%.4f", ff.bottleneckCap / maxCapacitySeen));
            sb.append("]}");
        }
        sb.append("],");
        
        // Links JSON
        sb.append("\"links\":[");
        PhysicalTopology topo = nos.getPhysicalTopology();
        int linkCount = 0;
        double maxUtil = 0;
        if (topo != null) {
            Collection<Link> links = topo.getAllLinks();
            first = true;
            for (Link link : links) {
                if (!first) sb.append(",");
                first = false;
                linkCount++;
                
                Node highNode = link.getHighOrder();
                Node lowNode = link.getLowOrder();
                
                double upBw = link.getBw(lowNode);
                double upFreeBw = link.getFreeBandwidth(lowNode);
                double upUtil = upBw > 0 ? (upBw - upFreeBw) / upBw : 0;
                int upChannels = link.getChannelCount(lowNode);
                
                double downBw = link.getBw(highNode);
                double downFreeBw = link.getFreeBandwidth(highNode);
                double downUtil = downBw > 0 ? (downBw - downFreeBw) / downBw : 0;
                int downChannels = link.getChannelCount(highNode);
                
                double linkMaxUtil = Math.max(upUtil, downUtil);
                if (linkMaxUtil > maxUtil) maxUtil = linkMaxUtil;
                
                sb.append("{");
                sb.append("\"up_bw\":").append((long) upBw).append(",");
                sb.append("\"down_bw\":").append((long) downBw).append(",");
                sb.append("\"up_util\":").append(String.format("%.4f", upUtil)).append(",");
                sb.append("\"down_util\":").append(String.format("%.4f", downUtil)).append(",");
                sb.append("\"channels\":").append(upChannels + downChannels);
                sb.append("}");
            }
        }
        sb.append("],");
        
        CFRRLLogger.debug(LOG_TAG, "Links: " + linkCount + ", Max utilization: " + String.format("%.2f%%", maxUtil * 100));
        
        sb.append("\"total_flows\":").append(flowFeaturesList.size()).append(",");
        sb.append("\"total_channels\":").append(nos.getChannelManager().getTotalChannelNum());
        sb.append("}");
        
        return sb.toString();
    }

    private List<FlowFeatures> extractFlowFeatures() {
        List<FlowFeatures> features = new ArrayList<>();
        ChannelManager cm = nos.getChannelManager();
        
        Map<String, Integer> vmNameToId = NetworkOperatingSystem.getVmNameToIdMap();
        Set<Integer> processedFlows = new HashSet<>();
        
        CFRRLLogger.debug(LOG_TAG, "Known VMs: " + vmNameToId.size());
        
        for (Integer vmId : vmNameToId.values()) {
            if (vmId == null || vmId == -1) continue;
            
            List<Channel> channels = cm.findAllChannels(vmId);
            
            for (Channel ch : channels) {
                int flowId = ch.getChId();
                if (flowId == -1) continue;
                if (processedFlows.contains(flowId)) continue;
                processedFlows.add(flowId);
                
                FlowFeatures ff = new FlowFeatures();
                ff.flowId = flowId;
                ff.srcVm = ch.getSrcId();
                ff.dstVm = ch.getDstId();
                ff.demand = ch.getRequestedBandwidth();
                
                if (ff.demand > maxDemandSeen) {
                    maxDemandSeen = ff.demand;
                }
                
                ff.pathLength = getPathLength(ch);
                ff.bottleneckCap = getBottleneckCapacity(ch);
                ff.numPaths = estimateNumPaths(ff.srcVm, ff.dstVm);
                
                if (ff.bottleneckCap > maxCapacitySeen) {
                    maxCapacitySeen = ff.bottleneckCap;
                }
                
                features.add(ff);
            }
        }
        
        return features;
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
    
    private int estimateNumPaths(int srcVm, int dstVm) {
        try {
            SDNHost srcHost = nos.findHost(srcVm);
            SDNHost dstHost = nos.findHost(dstVm);
            
            if (srcHost == null || dstHost == null) return 1;
            if (srcHost.equals(dstHost)) return 1;
            
            List<Link> candidates = srcHost.getRoute(dstHost);
            if (candidates != null) {
                return Math.max(1, candidates.size());
            }
        } catch (Exception e) {
            // Ignore
        }
        return 2;
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
            Map<String, Integer> flowNameToId = NetworkOperatingSystem.getFlowNameToIdMap();
            
            for (String item : items) {
                item = item.trim().replace("\"", "");
                if (item.isEmpty()) continue;
                
                try {
                    int flowId = Integer.parseInt(item);
                    criticalFlows.add(flowId);
                    CFRRLLogger.debug(LOG_TAG, "Parsed flow ID: " + flowId);
                } catch (NumberFormatException e) {
                    Integer flowId = flowNameToId.get(item);
                    if (flowId != null && flowId != -1) {
                        criticalFlows.add(flowId);
                        CFRRLLogger.debug(LOG_TAG, "Resolved flow name '" + item + "' to ID: " + flowId);
                    } else {
                        CFRRLLogger.warn(LOG_TAG, "Unknown flow identifier: " + item);
                    }
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

    private static class FlowFeatures {
        int flowId;
        int srcVm;
        int dstVm;
        double demand;
        int pathLength;
        double bottleneckCap;
        int numPaths;
    }
}