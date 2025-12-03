package org.cloudbus.cloudsim.sdn.rl;

import java.io.*;
import java.util.*;

/**
 * LatencyCollector: Centralized collection of network latency metrics.
 * 
 * PURPOSE:
 * ========
 * The original CFR-RL optimizes for Maximum Link Utilization (MLU) only.
 * We want to measure whether MLU-optimized routing inadvertently hurts latency.
 * This class collects per-packet latency data to enable that comparison.
 * 
 * HOW LATENCY IS COMPUTED:
 * ========================
 * When a packet travels through the network, it experiences three types of delay:
 * 
 * 1. PROPAGATION DELAY: Time for signal to travel physical distance
 *    - Fixed per path, based on link lengths
 *    - Retrieved from Channel.getTotalLatency()
 * 
 * 2. TRANSMISSION DELAY: Time to push bits onto the wire
 *    - = packet_size / bandwidth
 *    - This is the "ideal" transfer time with no congestion
 * 
 * 3. QUEUING DELAY: Time spent waiting behind other packets
 *    - = serve_time - propagation_delay - transmission_delay
 *    - This is what we care about! It reflects congestion.
 *    - MLU-optimized routing may concentrate traffic, increasing queuing.
 * 
 * USAGE:
 * ======
 * SDNDatacenter calls recordPacketCompletion() when each packet finishes.
 * At each RL interval, the agent calls getIntervalAvgQueuingDelayAndReset().
 * At simulation end, call printSummary() and export results.
 * 
 * @author CFR-RL Latency Research
 */
public class LatencyCollector {
    
    // =========================================================================
    // SINGLETON PATTERN
    // We use a singleton because latency recording happens across the entire
    // simulation, and we need a single point of aggregation.
    // =========================================================================
    
    private static LatencyCollector instance;
    
    public static synchronized LatencyCollector getInstance() {
        if (instance == null) {
            instance = new LatencyCollector();
        }
        return instance;
    }
    
    /**
     * Reset for a new simulation run.
     * Call this at the start of each experiment.
     */
    public static synchronized void reset() {
        if (instance != null) {
            instance.close();
        }
        instance = new LatencyCollector();
        CFRRLLogger.info("LatencyCollector", "Reset for new simulation");
    }
    
    private LatencyCollector() {
        CFRRLLogger.info("LatencyCollector", "Initialized");
    }
    
    // =========================================================================
    // INTERVAL STATISTICS
    // These are reset at each RL decision interval (e.g., every 5 seconds).
    // The Python agent uses these to see latency performance of current routing.
    // =========================================================================
    
    private double intervalQueuingDelaySum = 0;      // Sum of queuing delays this interval
    private double intervalServeTimeSum = 0;          // Sum of total serve times
    private int intervalPacketCount = 0;              // Number of packets this interval
    private double intervalMaxQueuingDelay = 0;       // Worst-case queuing this interval
    private List<Double> intervalQueuingDelays = new ArrayList<>();  // For percentile calculation
    
    // =========================================================================
    // GLOBAL STATISTICS
    // These accumulate across the entire simulation for final reporting.
    // =========================================================================
    
    private long totalPacketsCompleted = 0;
    private long totalPacketsFailed = 0;
    private double globalQueuingDelaySum = 0;
    private double globalMaxQueuingDelay = 0;
    
    // History of interval averages (for trend analysis)
    private List<Double> intervalAvgHistory = new ArrayList<>();
    
    // =========================================================================
    // PER-FLOW TRACKING
    // Track latency by flow to see if critical flows get better/worse treatment.
    // =========================================================================
    
    private Map<Integer, FlowStats> flowStatsMap = new HashMap<>();
    
    // Which flows are currently "critical" (being RL-optimized)
    private Set<Integer> criticalFlows = new HashSet<>();
    
    // =========================================================================
    // CSV EXPORT
    // Write every packet to CSV for offline analysis and plotting.
    // =========================================================================
    
    private PrintWriter csvWriter = null;
    private String csvFilePath = null;
    
    // =========================================================================
    // CORE RECORDING METHOD
    // Called by SDNDatacenter.processPacketCompleted() for every packet.
    // =========================================================================
    
    /**
     * Record a completed packet transmission.
     * 
     * This is the main entry point, called from SDNDatacenter.
     * 
     * @param flowId          Which flow this packet belongs to
     * @param packetId        Unique packet identifier (for debugging)
     * @param startTime       When packet entered network (from Packet.getStartTime())
     * @param finishTime      When packet finished (from Packet.getFinishTime())
     * @param propagationDelay Physical delay from Channel.getTotalLatency()
     * @param packetSize      Size in bytes
     * @param bandwidth       Allocated bandwidth from Channel.getAllocatedBandwidth()
     * @param pathLength      Number of hops from Channel.getPathLength()
     * @param srcNode         Source node ID (for logging)
     * @param dstNode         Destination node ID (for logging)
     */
    public synchronized void recordPacketCompletion(
            int flowId,
            long packetId,
            double startTime,
            double finishTime,
            double propagationDelay,
            long packetSize,
            double bandwidth,
            int pathLength,
            int srcNode,
            int dstNode) {
        
        // =====================================================================
        // COMPUTE LATENCY COMPONENTS
        // =====================================================================
        
        // Total time packet spent in network
        double serveTime = finishTime - startTime;
        
        // Time to transmit the bits (ideal, no congestion)
        double transmissionDelay = (bandwidth > 0) ? (double) packetSize / bandwidth : 0;
        
        // QUEUING DELAY = what's left after accounting for physics
        // This is the congestion-induced delay we want to minimize!
        double queuingDelay = Math.max(0, serveTime - propagationDelay - transmissionDelay);
        
        // =====================================================================
        // UPDATE INTERVAL STATISTICS
        // =====================================================================
        
        intervalQueuingDelaySum += queuingDelay;
        intervalServeTimeSum += serveTime;
        intervalPacketCount++;
        intervalMaxQueuingDelay = Math.max(intervalMaxQueuingDelay, queuingDelay);
        intervalQueuingDelays.add(queuingDelay);
        
        // =====================================================================
        // UPDATE GLOBAL STATISTICS
        // =====================================================================
        
        totalPacketsCompleted++;
        globalQueuingDelaySum += queuingDelay;
        globalMaxQueuingDelay = Math.max(globalMaxQueuingDelay, queuingDelay);
        
        // =====================================================================
        // UPDATE PER-FLOW STATISTICS
        // =====================================================================
        
        FlowStats fs = flowStatsMap.computeIfAbsent(flowId, k -> new FlowStats(flowId));
        fs.addPacket(queuingDelay, serveTime);
        
        // =====================================================================
        // WRITE TO CSV (if enabled)
        // =====================================================================
        
        if (csvWriter != null) {
            boolean isCritical = criticalFlows.contains(flowId);
            csvWriter.printf("%.6f,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%.0f,%d,%d,%d,%b%n",
                finishTime, flowId, packetId, startTime, finishTime, serveTime,
                propagationDelay, transmissionDelay, queuingDelay, packetSize,
                bandwidth, pathLength, srcNode, dstNode, isCritical);
            csvWriter.flush();
        }
        
        // =====================================================================
        // LOG HIGH QUEUING DELAYS (for debugging)
        // =====================================================================
        
        if (queuingDelay > 0.1) { // > 100ms is notable
            CFRRLLogger.debug("LatencyCollector", String.format(
                "High queuing: flow=%d, queue=%.4fs, serve=%.4fs, path=%d hops",
                flowId, queuingDelay, serveTime, pathLength));
        }
    }
    
    /**
     * Record a failed/timed-out packet.
     * Failed packets indicate severe congestion.
     */
    public synchronized void recordPacketFailure(
            int flowId,
            long packetId,
            double startTime,
            double failTime,
            int srcNode,
            int dstNode) {
        
        totalPacketsFailed++;
        
        CFRRLLogger.warn("LatencyCollector", String.format(
            "Packet FAILED: flow=%d, src=%d->dst=%d, waited=%.4fs",
            flowId, srcNode, dstNode, failTime - startTime));
    }
    
    // =========================================================================
    // CRITICAL FLOW MANAGEMENT
    // The RL agent selects which flows to optimize. Track these separately.
    // =========================================================================
    
    /**
     * Update which flows are currently being optimized by the RL agent.
     */
    public synchronized void setCriticalFlows(Set<Integer> flows) {
        this.criticalFlows = new HashSet<>(flows);
        CFRRLLogger.info("LatencyCollector", "Critical flows updated: " + flows.size() + " flows");
    }
    
    // =========================================================================
    // INTERVAL QUERY METHODS
    // Called by the RL agent at each decision interval.
    // =========================================================================
    
    /**
     * Get average queuing delay for this interval, then reset for next interval.
     * 
     * This is the KEY METHOD for the RL agent. It returns the latency performance
     * of the current routing, which can be used in the reward function.
     */
    public synchronized double getIntervalAvgQueuingDelayAndReset() {
        double avg = intervalPacketCount > 0 ? intervalQueuingDelaySum / intervalPacketCount : 0;
        
        // Log interval summary
        CFRRLLogger.info("LatencyCollector", String.format(
            "Interval: packets=%d, avgQueuing=%.6fs, maxQueuing=%.6fs",
            intervalPacketCount, avg, intervalMaxQueuingDelay));
        
        // Save to history
        intervalAvgHistory.add(avg);
        
        // Reset interval counters
        intervalQueuingDelaySum = 0;
        intervalServeTimeSum = 0;
        intervalPacketCount = 0;
        intervalMaxQueuingDelay = 0;
        intervalQueuingDelays.clear();
        
        return avg;
    }
    
    /**
     * Get detailed interval statistics without resetting.
     * Useful for reporting percentiles.
     */
    public synchronized IntervalStats getIntervalStats() {
        IntervalStats stats = new IntervalStats();
        stats.packetCount = intervalPacketCount;
        stats.avgQueuingDelay = intervalPacketCount > 0 ? intervalQueuingDelaySum / intervalPacketCount : 0;
        stats.maxQueuingDelay = intervalMaxQueuingDelay;
        
        // Calculate percentiles
        if (!intervalQueuingDelays.isEmpty()) {
            List<Double> sorted = new ArrayList<>(intervalQueuingDelays);
            Collections.sort(sorted);
            int n = sorted.size();
            stats.p50QueuingDelay = sorted.get(n / 2);
            stats.p95QueuingDelay = sorted.get(Math.min(n - 1, (int)(n * 0.95)));
            stats.p99QueuingDelay = sorted.get(Math.min(n - 1, (int)(n * 0.99)));
        }
        
        return stats;
    }
    
    // =========================================================================
    // GLOBAL QUERY METHODS
    // For final reporting at end of simulation.
    // =========================================================================
    
    /**
     * Get aggregate statistics for the entire simulation.
     */
    public synchronized GlobalStats getGlobalStats() {
        GlobalStats stats = new GlobalStats();
        stats.totalPacketsCompleted = totalPacketsCompleted;
        stats.totalPacketsFailed = totalPacketsFailed;
        stats.globalAvgQueuingDelay = totalPacketsCompleted > 0 ? 
            globalQueuingDelaySum / totalPacketsCompleted : 0;
        stats.globalMaxQueuingDelay = globalMaxQueuingDelay;
        stats.numFlows = flowStatsMap.size();
        stats.numIntervals = intervalAvgHistory.size();
        return stats;
    }
    
    /**
     * Get per-flow statistics.
     * Useful for comparing critical vs background flow latency.
     */
    public synchronized Map<Integer, FlowStats> getFlowStats() {
        return new HashMap<>(flowStatsMap);
    }
    
    // =========================================================================
    // JSON EXPORT (for Python agent communication)
    // =========================================================================
    
    /**
     * Export interval stats as JSON for Python agent.
     */
    public synchronized String toIntervalJson() {
        IntervalStats stats = getIntervalStats();
        StringBuilder sb = new StringBuilder();
        sb.append("{");
        sb.append("\"packet_count\":").append(stats.packetCount).append(",");
        sb.append("\"avg_queuing_delay\":").append(stats.avgQueuingDelay).append(",");
        sb.append("\"max_queuing_delay\":").append(stats.maxQueuingDelay).append(",");
        sb.append("\"p50_queuing_delay\":").append(stats.p50QueuingDelay).append(",");
        sb.append("\"p95_queuing_delay\":").append(stats.p95QueuingDelay).append(",");
        sb.append("\"p99_queuing_delay\":").append(stats.p99QueuingDelay);
        sb.append("}");
        return sb.toString();
    }
    
    // =========================================================================
    // CSV EXPORT
    // =========================================================================
    
    /**
     * Start writing per-packet records to CSV.
     * Call this at the start of simulation if you want detailed data.
     */
    public synchronized void initCsvExport(String path) {
        this.csvFilePath = path;
        try {
            csvWriter = new PrintWriter(new BufferedWriter(new FileWriter(path, false)));
            // Write header
            csvWriter.println("timestamp,flow_id,packet_id,start_time,finish_time,serve_time," +
                            "propagation_delay,transmission_delay,queuing_delay,packet_size," +
                            "bandwidth,path_length,src_node,dst_node,is_critical");
            csvWriter.flush();
            CFRRLLogger.info("LatencyCollector", "CSV export initialized: " + path);
        } catch (IOException e) {
            CFRRLLogger.error("LatencyCollector", "Failed to init CSV: " + e.getMessage());
            csvWriter = null;
        }
    }
    
    // =========================================================================
    // SUMMARY REPORTING
    // =========================================================================
    
    /**
     * Print comprehensive summary to logger.
     * Call at end of simulation.
     */
    public synchronized void printSummary() {
        GlobalStats global = getGlobalStats();
        
        CFRRLLogger.section("LATENCY STATISTICS SUMMARY");
        
        CFRRLLogger.info("LatencyCollector", "=== Global Statistics ===");
        CFRRLLogger.info("LatencyCollector", "  Total packets: " + global.totalPacketsCompleted);
        CFRRLLogger.info("LatencyCollector", "  Failed packets: " + global.totalPacketsFailed);
        CFRRLLogger.info("LatencyCollector", String.format("  Avg queuing delay: %.6f s", global.globalAvgQueuingDelay));
        CFRRLLogger.info("LatencyCollector", String.format("  Max queuing delay: %.6f s", global.globalMaxQueuingDelay));
        
        // Critical vs background comparison
        int criticalCount = 0, backgroundCount = 0;
        double criticalSum = 0, backgroundSum = 0;
        
        for (FlowStats fs : flowStatsMap.values()) {
            if (criticalFlows.contains(fs.flowId)) {
                criticalCount += fs.packetCount;
                criticalSum += fs.queuingDelaySum;
            } else {
                backgroundCount += fs.packetCount;
                backgroundSum += fs.queuingDelaySum;
            }
        }
        
        CFRRLLogger.info("LatencyCollector", "=== Critical vs Background ===");
        if (criticalCount > 0) {
            CFRRLLogger.info("LatencyCollector", String.format(
                "  Critical flows: %d packets, avg queuing = %.6f s",
                criticalCount, criticalSum / criticalCount));
        }
        if (backgroundCount > 0) {
            CFRRLLogger.info("LatencyCollector", String.format(
                "  Background flows: %d packets, avg queuing = %.6f s",
                backgroundCount, backgroundSum / backgroundCount));
        }
    }
    
    /**
     * Close resources.
     */
    public synchronized void close() {
        if (csvWriter != null) {
            csvWriter.close();
            csvWriter = null;
            CFRRLLogger.info("LatencyCollector", "CSV closed: " + csvFilePath);
        }
    }
    
    // =========================================================================
    // HELPER DATA CLASSES
    // =========================================================================
    
    /** Statistics for a single RL interval. */
    public static class IntervalStats {
        public int packetCount;
        public double avgQueuingDelay;
        public double maxQueuingDelay;
        public double p50QueuingDelay;
        public double p95QueuingDelay;
        public double p99QueuingDelay;
    }
    
    /** Aggregate statistics for entire simulation. */
    public static class GlobalStats {
        public long totalPacketsCompleted;
        public long totalPacketsFailed;
        public double globalAvgQueuingDelay;
        public double globalMaxQueuingDelay;
        public int numFlows;
        public int numIntervals;
    }
    
    /** Per-flow statistics. */
    public static class FlowStats {
        public final int flowId;
        public int packetCount = 0;
        public double queuingDelaySum = 0;
        public double maxQueuingDelay = 0;
        
        public FlowStats(int flowId) {
            this.flowId = flowId;
        }
        
        public void addPacket(double queuingDelay, double serveTime) {
            packetCount++;
            queuingDelaySum += queuingDelay;
            maxQueuingDelay = Math.max(maxQueuingDelay, queuingDelay);
        }
        
        public double getAvgQueuingDelay() {
            return packetCount > 0 ? queuingDelaySum / packetCount : 0;
        }
    }
}