package org.cloudbus.cloudsim.sdn.rl;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.ArrayList;
import java.util.List;

/**
 * LinkStatsCollector - Tracks per-link utilization statistics for RL training.
 * 
 * Unlike the built-in link_utilization_*.csv which logs per-node,
 * this collector tracks actual link-level statistics with proper link IDs.
 * 
 * Tracked metrics per link:
 * - Total bytes transferred
 * - Average utilization over episode
 * - Maximum utilization (for bottleneck detection)
 * - Congestion events (times utilization > threshold)
 * 
 * Usage:
 *   // At simulation start
 *   LinkStatsCollector.getInstance().reset();
 *   
 *   // During simulation (called from Link.updateMonitor or separately)
 *   LinkStatsCollector.getInstance().recordLinkUtilization(linkId, utilization, timestamp);
 *   
 *   // At simulation end
 *   LinkStatsCollector.getInstance().exportToCsv("outputs/link_stats.csv");
 *   Map<String, LinkStats> stats = LinkStatsCollector.getInstance().getAllStats();
 * 
 * @author CFR-RL Extension
 */
public class LinkStatsCollector {
    
    // Singleton instance
    private static LinkStatsCollector instance = null;
    
    // Congestion threshold (utilization > this triggers congestion event)
    private static final double CONGESTION_THRESHOLD = 0.9;  // 90%
    
    // Per-link statistics
    private Map<String, LinkStats> linkStatsMap;
    
    // Episode tracking
    private double episodeStartTime = 0;
    private double lastUpdateTime = 0;
    
    /**
     * Statistics container for a single link
     */
    public static class LinkStats {
        public String linkId;
        public String srcNode;
        public String dstNode;
        public double capacity;         // Link capacity in bps
        
        // Utilization tracking
        public double totalUtilization = 0;  // Sum of utilization samples
        public int sampleCount = 0;          // Number of samples
        public double maxUtilization = 0;    // Peak utilization
        public double minUtilization = 1.0;  // Minimum utilization
        
        // Byte tracking
        public long totalBytes = 0;          // Total bytes transferred
        
        // Congestion events
        public int congestionEvents = 0;     // Times utilization > threshold
        public double timeInCongestion = 0;  // Total time spent congested
        
        // Temporal tracking
        public double lastUtilization = 0;
        public double lastTimestamp = 0;
        public List<Double> utilizationHistory = new ArrayList<>();  // For detailed analysis
        
        public LinkStats(String linkId, String srcNode, String dstNode, double capacity) {
            this.linkId = linkId;
            this.srcNode = srcNode;
            this.dstNode = dstNode;
            this.capacity = capacity;
        }
        
        public double getAverageUtilization() {
            return sampleCount > 0 ? totalUtilization / sampleCount : 0;
        }
        
        public double getUtilizationStdDev() {
            if (sampleCount < 2) return 0;
            double mean = getAverageUtilization();
            double sumSquaredDiff = 0;
            for (double u : utilizationHistory) {
                sumSquaredDiff += (u - mean) * (u - mean);
            }
            return Math.sqrt(sumSquaredDiff / sampleCount);
        }
    }
    
    private LinkStatsCollector() {
        linkStatsMap = new HashMap<>();
    }
    
    public static synchronized LinkStatsCollector getInstance() {
        if (instance == null) {
            instance = new LinkStatsCollector();
        }
        return instance;
    }
    
    /**
     * Reset all statistics for a new episode
     */
    public void reset() {
        linkStatsMap.clear();
        episodeStartTime = 0;
        lastUpdateTime = 0;
        CFRRLLogger.info("LinkStatsCollector", "Reset for new episode");
    }
    
    /**
     * Initialize a link for tracking
     * Call this at simulation setup for each link in the topology
     */
    public void registerLink(String srcNode, String dstNode, double capacity) {
        String linkId = getLinkId(srcNode, dstNode);
        if (!linkStatsMap.containsKey(linkId)) {
            linkStatsMap.put(linkId, new LinkStats(linkId, srcNode, dstNode, capacity));
            CFRRLLogger.debug("LinkStatsCollector", "Registered link: " + linkId + " (capacity=" + capacity + ")");
        }
    }
    
    /**
     * Generate a unique link ID from node names
     * Uses alphabetical ordering for consistency (A→B same as B→A for undirected)
     * But we track direction, so we use src→dst format
     */
    public static String getLinkId(String srcNode, String dstNode) {
        return srcNode + "->" + dstNode;
    }
    
    /**
     * Record utilization sample for a link
     * 
     * @param srcNode    Source node name
     * @param dstNode    Destination node name  
     * @param utilization Current utilization [0, 1]
     * @param timestamp   Simulation time
     * @param bytesTransferred Bytes transferred this period
     */
    public void recordLinkUtilization(String srcNode, String dstNode, 
                                       double utilization, double timestamp,
                                       long bytesTransferred) {
        String linkId = getLinkId(srcNode, dstNode);
        LinkStats stats = linkStatsMap.get(linkId);
        
        if (stats == null) {
            // Auto-register if not registered (shouldn't happen normally)
            CFRRLLogger.warn("LinkStatsCollector", "Auto-registering unknown link: " + linkId);
            stats = new LinkStats(linkId, srcNode, dstNode, 0);
            linkStatsMap.put(linkId, stats);
        }
        
        // Update statistics
        stats.totalUtilization += utilization;
        stats.sampleCount++;
        stats.maxUtilization = Math.max(stats.maxUtilization, utilization);
        stats.minUtilization = Math.min(stats.minUtilization, utilization);
        stats.totalBytes += bytesTransferred;
        stats.utilizationHistory.add(utilization);
        
        // Check for congestion event
        if (utilization > CONGESTION_THRESHOLD) {
            // Only count as new event if we weren't already congested
            if (stats.lastUtilization <= CONGESTION_THRESHOLD) {
                stats.congestionEvents++;
            }
            // Track time in congestion
            if (stats.lastTimestamp > 0) {
                stats.timeInCongestion += (timestamp - stats.lastTimestamp);
            }
        }
        
        stats.lastUtilization = utilization;
        stats.lastTimestamp = timestamp;
        lastUpdateTime = timestamp;
        
        // Set episode start time on first update
        if (episodeStartTime == 0) {
            episodeStartTime = timestamp;
        }
    }
    
    /**
     * Get statistics for a specific link
     */
    public LinkStats getLinkStats(String srcNode, String dstNode) {
        return linkStatsMap.get(getLinkId(srcNode, dstNode));
    }
    
    /**
     * Get all link statistics
     */
    public Map<String, LinkStats> getAllStats() {
        return linkStatsMap;
    }
    
    /**
     * Get maximum utilization across all links (for global bottleneck)
     */
    public double getGlobalMaxUtilization() {
        double maxUtil = 0;
        for (LinkStats stats : linkStatsMap.values()) {
            maxUtil = Math.max(maxUtil, stats.maxUtilization);
        }
        return maxUtil;
    }
    
    /**
     * Get the most congested link
     */
    public LinkStats getMostCongestedLink() {
        LinkStats worst = null;
        double worstUtil = 0;
        for (LinkStats stats : linkStatsMap.values()) {
            if (stats.maxUtilization > worstUtil) {
                worstUtil = stats.maxUtilization;
                worst = stats;
            }
        }
        return worst;
    }
    
    /**
     * Get utilization statistics for links on a path
     * 
     * @param pathNodes List of node names on the path
     * @return Map with avg_util, max_util, congestion_events for the path
     */
    public Map<String, Double> getPathStats(List<String> pathNodes) {
        Map<String, Double> result = new HashMap<>();
        double sumUtil = 0;
        double maxUtil = 0;
        int totalCongestionEvents = 0;
        int linkCount = 0;
        
        // Iterate through consecutive node pairs
        for (int i = 0; i < pathNodes.size() - 1; i++) {
            String src = pathNodes.get(i);
            String dst = pathNodes.get(i + 1);
            LinkStats stats = getLinkStats(src, dst);
            
            if (stats != null && stats.sampleCount > 0) {
                sumUtil += stats.getAverageUtilization();
                maxUtil = Math.max(maxUtil, stats.maxUtilization);
                totalCongestionEvents += stats.congestionEvents;
                linkCount++;
            }
        }
        
        result.put("avg_utilization", linkCount > 0 ? sumUtil / linkCount : 0);
        result.put("max_utilization", maxUtil);
        result.put("congestion_events", (double) totalCongestionEvents);
        result.put("link_count", (double) linkCount);
        
        return result;
    }
    
    /**
     * Export statistics to CSV file
     */
    public void exportToCsv(String filename) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filename))) {
            // Header
            writer.println("link_id,src_node,dst_node,capacity,total_bytes," +
                          "avg_utilization,max_utilization,min_utilization,std_utilization," +
                          "sample_count,congestion_events,time_in_congestion");
            
            // Data rows
            for (LinkStats stats : linkStatsMap.values()) {
                writer.println(String.format("%s,%s,%s,%.0f,%d,%.6f,%.6f,%.6f,%.6f,%d,%d,%.4f",
                    stats.linkId,
                    stats.srcNode,
                    stats.dstNode,
                    stats.capacity,
                    stats.totalBytes,
                    stats.getAverageUtilization(),
                    stats.maxUtilization,
                    stats.minUtilization,
                    stats.getUtilizationStdDev(),
                    stats.sampleCount,
                    stats.congestionEvents,
                    stats.timeInCongestion
                ));
            }
            
            CFRRLLogger.info("LinkStatsCollector", "Exported " + linkStatsMap.size() + 
                           " link stats to " + filename);
            
        } catch (IOException e) {
            CFRRLLogger.error("LinkStatsCollector", "Failed to export CSV: " + e.getMessage());
        }
    }
    
    /**
     * Print summary to log
     */
    public void printSummary() {
        CFRRLLogger.info("LinkStatsCollector", "========== Link Statistics Summary ==========");
        CFRRLLogger.info("LinkStatsCollector", "Total links tracked: " + linkStatsMap.size());
        CFRRLLogger.info("LinkStatsCollector", "Episode duration: " + 
                        String.format("%.2f", lastUpdateTime - episodeStartTime) + "s");
        
        // Find bottleneck
        LinkStats bottleneck = getMostCongestedLink();
        if (bottleneck != null) {
            CFRRLLogger.info("LinkStatsCollector", "Bottleneck link: " + bottleneck.linkId + 
                           " (max_util=" + String.format("%.2f%%", bottleneck.maxUtilization * 100) + ")");
        }
        
        // Count congested links
        int congestedLinks = 0;
        int totalCongestionEvents = 0;
        for (LinkStats stats : linkStatsMap.values()) {
            if (stats.maxUtilization > CONGESTION_THRESHOLD) {
                congestedLinks++;
            }
            totalCongestionEvents += stats.congestionEvents;
        }
        CFRRLLogger.info("LinkStatsCollector", "Links that exceeded " + 
                        (CONGESTION_THRESHOLD * 100) + "% utilization: " + congestedLinks);
        CFRRLLogger.info("LinkStatsCollector", "Total congestion events: " + totalCongestionEvents);
        
        // Top 5 most utilized links
        CFRRLLogger.info("LinkStatsCollector", "Top 5 most utilized links:");
        linkStatsMap.values().stream()
            .sorted((a, b) -> Double.compare(b.maxUtilization, a.maxUtilization))
            .limit(5)
            .forEach(stats -> {
                CFRRLLogger.info("LinkStatsCollector", String.format("  %s: avg=%.2f%%, max=%.2f%%, events=%d",
                    stats.linkId, 
                    stats.getAverageUtilization() * 100,
                    stats.maxUtilization * 100,
                    stats.congestionEvents));
            });
    }
    
    /**
     * Generate episode summary as a map (for JSON export or Python consumption)
     */
    public Map<String, Object> generateEpisodeSummary() {
        Map<String, Object> summary = new HashMap<>();
        
        summary.put("total_links", linkStatsMap.size());
        summary.put("episode_duration", lastUpdateTime - episodeStartTime);
        summary.put("global_max_utilization", getGlobalMaxUtilization());
        
        // Congestion stats
        int congestedLinks = 0;
        int totalCongestionEvents = 0;
        double totalTimeInCongestion = 0;
        for (LinkStats stats : linkStatsMap.values()) {
            if (stats.maxUtilization > CONGESTION_THRESHOLD) {
                congestedLinks++;
            }
            totalCongestionEvents += stats.congestionEvents;
            totalTimeInCongestion += stats.timeInCongestion;
        }
        
        summary.put("congested_link_count", congestedLinks);
        summary.put("total_congestion_events", totalCongestionEvents);
        summary.put("total_time_in_congestion", totalTimeInCongestion);
        
        // Bottleneck info
        LinkStats bottleneck = getMostCongestedLink();
        if (bottleneck != null) {
            summary.put("bottleneck_link", bottleneck.linkId);
            summary.put("bottleneck_max_util", bottleneck.maxUtilization);
        }
        
        return summary;
    }
}