package org.cloudbus.cloudsim.sdn.policies.selectlink;

import java.util.*;

import org.cloudbus.cloudsim.sdn.physicalcomponents.Link;
import org.cloudbus.cloudsim.sdn.physicalcomponents.Node;
import org.cloudbus.cloudsim.sdn.rl.CFRRLLogger;

/**
 * CFR-RL Link Selection Policy
 * 
 * - Critical flows: bandwidth-aware routing (least congested path)
 * - Background flows: hash-based routing (fast, deterministic)
 * 
 * With comprehensive logging for debugging.
 */
public class LinkSelectionPolicyCFRRL implements LinkSelectionPolicy {
    
    private static final String LOG_TAG = "LinkPolicy";
    
    private Set<Integer> criticalFlowIds;
    private int kCritical;
    
    // Statistics
    private long criticalRoutingCalls = 0;
    private long backgroundRoutingCalls = 0;
    private long totalSelectLinkCalls = 0;
    
    // Detailed logging control
    private boolean verboseLogging = true;
    private int logEveryN = 100;  // Log every Nth call to reduce spam
    
    public LinkSelectionPolicyCFRRL() {
        this(8);
    }
    
    public LinkSelectionPolicyCFRRL(int kCritical) {
        this.kCritical = kCritical;
        this.criticalFlowIds = new HashSet<>();
        CFRRLLogger.info(LOG_TAG, "Created with K=" + kCritical);
    }
    
    /**
     * Enable/disable verbose per-call logging.
     */
    public void setVerboseLogging(boolean verbose) {
        this.verboseLogging = verbose;
        CFRRLLogger.info(LOG_TAG, "Verbose logging: " + verbose);
    }
    
    @Override
    public Link selectLink(List<Link> links, int flowId, Node srcHost, Node destHost, Node prevNode) {
        totalSelectLinkCalls++;
        
        // Log periodically to avoid spam
        boolean shouldLog = verboseLogging && (totalSelectLinkCalls % logEveryN == 1);
        
        if (shouldLog) {
            CFRRLLogger.debug(LOG_TAG, String.format(
                "selectLink called #%d: flowId=%d, prevNode=%s, numLinks=%d",
                totalSelectLinkCalls, flowId, prevNode, links != null ? links.size() : 0));
        }
        
        // Single link - no choice
        if (links == null || links.size() <= 1) {
            if (shouldLog) {
                CFRRLLogger.debug(LOG_TAG, "Single link available, returning it");
            }
            return links != null && links.size() == 1 ? links.get(0) : null;
        }
        
        // Check if critical
        boolean isCritical = flowId != -1 && criticalFlowIds.contains(flowId);
        
        Link selected;
        if (isCritical) {
            criticalRoutingCalls++;
            selected = selectLinkForCriticalFlow(links, flowId, srcHost, destHost, prevNode, shouldLog);
        } else {
            backgroundRoutingCalls++;
            selected = selectLinkForBackgroundFlow(links, flowId, srcHost, destHost, prevNode, shouldLog);
        }
        
        if (shouldLog) {
            CFRRLLogger.debug(LOG_TAG, String.format(
                "Selected link for flow %d (%s): %s",
                flowId, isCritical ? "CRITICAL" : "background", selected));
        }
        
        return selected;
    }
    
    private Link selectLinkForCriticalFlow(List<Link> links, int flowId, 
                                            Node srcHost, Node destHost, Node prevNode,
                                            boolean shouldLog) {
        Link bestLink = null;
        int minChannelCount = Integer.MAX_VALUE;
        
        if (shouldLog) {
            CFRRLLogger.debug(LOG_TAG, "Critical flow " + flowId + " - evaluating " + links.size() + " links:");
        }
        
        for (Link link : links) {
            int channelCount = link.getChannelCount(prevNode);
            
            if (shouldLog) {
                CFRRLLogger.debug(LOG_TAG, String.format("  Link %s: %d channels", link, channelCount));
            }
            
            if (channelCount < minChannelCount) {
                minChannelCount = channelCount;
                bestLink = link;
            }
        }
        
        if (shouldLog && links.size() > 1) {
            CFRRLLogger.debug(LOG_TAG, "Critical flow " + flowId + " -> best link has " + minChannelCount + " channels");
        }
        
        return bestLink != null ? bestLink : links.get(0);
    }
    
    private Link selectLinkForBackgroundFlow(List<Link> links, int flowId,
                                              Node srcHost, Node destHost, Node prevNode,
                                              boolean shouldLog) {
        int numLinks = links.size();
        int linkIndex = Math.abs(destHost.getAddress()) % numLinks;
        
        if (shouldLog) {
            CFRRLLogger.debug(LOG_TAG, String.format(
                "Background flow %d: destAddr=%d %% %d = index %d",
                flowId, destHost.getAddress(), numLinks, linkIndex));
        }
        
        return links.get(linkIndex);
    }
    
    /**
     * Update the set of critical flows.
     */
    public void setCriticalFlows(Set<Integer> newCriticalFlows) {
        Set<Integer> oldFlows = this.criticalFlowIds;
        this.criticalFlowIds = new HashSet<>(newCriticalFlows);
        
        // Log what changed
        Set<Integer> added = new HashSet<>(newCriticalFlows);
        added.removeAll(oldFlows);
        
        Set<Integer> removed = new HashSet<>(oldFlows);
        removed.removeAll(newCriticalFlows);
        
        CFRRLLogger.info(LOG_TAG, String.format(
            "Updated critical flows: %d total (added: %s, removed: %s)",
            criticalFlowIds.size(), added, removed));
    }
    
    public void addCriticalFlow(int flowId) {
        if (criticalFlowIds.size() < kCritical) {
            criticalFlowIds.add(flowId);
            CFRRLLogger.debug(LOG_TAG, "Added critical flow: " + flowId);
        }
    }
    
    public void removeCriticalFlow(int flowId) {
        criticalFlowIds.remove(flowId);
        CFRRLLogger.debug(LOG_TAG, "Removed critical flow: " + flowId);
    }
    
    public void clearCriticalFlows() {
        criticalFlowIds.clear();
        CFRRLLogger.debug(LOG_TAG, "Cleared all critical flows");
    }
    
    public boolean isCriticalFlow(int flowId) {
        return criticalFlowIds.contains(flowId);
    }
    
    public int getNumCriticalFlows() {
        return criticalFlowIds.size();
    }
    
    public int getKCritical() {
        return kCritical;
    }
    
    public void setKCritical(int k) {
        this.kCritical = k;
        CFRRLLogger.info(LOG_TAG, "K changed to: " + k);
    }
    
    public String getStatistics() {
        long total = criticalRoutingCalls + backgroundRoutingCalls;
        double criticalPercent = total > 0 ? (100.0 * criticalRoutingCalls / total) : 0;
        
        String stats = String.format(
            "Total selectLink calls: %d | Critical: %d (%.1f%%) | Background: %d (%.1f%%)",
            totalSelectLinkCalls,
            criticalRoutingCalls, criticalPercent,
            backgroundRoutingCalls, 100.0 - criticalPercent);
        
        return stats;
    }
    
    /**
     * Log detailed statistics.
     */
    public void logStatistics() {
        CFRRLLogger.info(LOG_TAG, "=== ROUTING STATISTICS ===");
        CFRRLLogger.info(LOG_TAG, "Total selectLink() calls: " + totalSelectLinkCalls);
        CFRRLLogger.info(LOG_TAG, "Critical flow routings:   " + criticalRoutingCalls);
        CFRRLLogger.info(LOG_TAG, "Background flow routings: " + backgroundRoutingCalls);
        CFRRLLogger.info(LOG_TAG, "Current critical flows:   " + criticalFlowIds);
        CFRRLLogger.info(LOG_TAG, "==========================");
    }
    
    @Override
    public boolean isDynamicRoutingEnabled() {
        return true;
    }
}