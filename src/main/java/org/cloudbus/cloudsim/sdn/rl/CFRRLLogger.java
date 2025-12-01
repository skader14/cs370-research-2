package org.cloudbus.cloudsim.sdn.rl;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * Centralized logging for CFR-RL components.
 * 
 * Logs to both console and file for easy debugging.
 * Default log file: cfrrl_debug.log (in working directory)
 * 
 * Usage:
 *   CFRRLLogger.init();  // Call once at startup
 *   CFRRLLogger.info("Component", "Message");
 *   CFRRLLogger.close(); // Call at shutdown
 */
public class CFRRLLogger {
    
    private static String logFilePath = "cfrrl_debug.log";
    private static PrintWriter fileWriter = null;
    private static boolean initialized = false;
    private static final SimpleDateFormat timeFormat = new SimpleDateFormat("HH:mm:ss.SSS");
    private static final SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
    
    // Log levels
    public static final int DEBUG = 0;
    public static final int INFO = 1;
    public static final int WARN = 2;
    public static final int ERROR = 3;
    
    private static int currentLevel = DEBUG;  // Log everything by default
    
    /**
     * Initialize the logger with default log file.
     */
    public static synchronized void init() {
        init(logFilePath);
    }
    
    /**
     * Initialize the logger with custom log file path.
     */
    public static synchronized void init(String path) {
        if (initialized) return;
        
        logFilePath = path;
        
        try {
            File logFile = new File(logFilePath);
            if (logFile.getParentFile() != null) {
                logFile.getParentFile().mkdirs();
            }
            
            fileWriter = new PrintWriter(new BufferedWriter(new FileWriter(logFilePath, false)));
            initialized = true;
            
            // Write header
            fileWriter.println("================================================================================");
            fileWriter.println("CFR-RL DEBUG LOG");
            fileWriter.println("Started: " + dateFormat.format(new Date()));
            fileWriter.println("Log file: " + logFile.getAbsolutePath());
            fileWriter.println("================================================================================");
            fileWriter.println();
            fileWriter.println("Format: [WallClock] [SimTime] [Level] [Component] Message");
            fileWriter.println();
            fileWriter.flush();
            
            System.out.println("[CFR-RL] ========================================");
            System.out.println("[CFR-RL] Logging initialized");
            System.out.println("[CFR-RL] Log file: " + logFile.getAbsolutePath());
            System.out.println("[CFR-RL] ========================================");
            
        } catch (IOException e) {
            System.err.println("[CFR-RL] WARNING: Could not create log file: " + e.getMessage());
            System.err.println("[CFR-RL] Logging to console only");
        }
    }
    
    /**
     * Set minimum log level (DEBUG=0, INFO=1, WARN=2, ERROR=3).
     */
    public static void setLevel(int level) {
        currentLevel = level;
        info("Logger", "Log level set to: " + levelToString(level));
    }
    
    private static String levelToString(int level) {
        switch (level) {
            case DEBUG: return "DEBUG";
            case INFO:  return "INFO";
            case WARN:  return "WARN";
            case ERROR: return "ERROR";
            default:    return "UNKNOWN";
        }
    }
    
    // ==================== Logging Methods ====================
    
    public static void debug(String component, String message) {
        log(DEBUG, component, message);
    }
    
    public static void info(String component, String message) {
        log(INFO, component, message);
    }
    
    public static void warn(String component, String message) {
        log(WARN, component, message);
    }
    
    public static void error(String component, String message) {
        log(ERROR, component, message);
    }
    
    public static void error(String component, String message, Throwable t) {
        log(ERROR, component, message + " - " + t.getClass().getSimpleName() + ": " + t.getMessage());
        if (fileWriter != null) {
            fileWriter.println("--- Stack Trace ---");
            t.printStackTrace(fileWriter);
            fileWriter.println("--- End Stack Trace ---");
            fileWriter.flush();
        }
        t.printStackTrace(System.err);
    }
    
    /**
     * Core logging method.
     */
    private static synchronized void log(int level, String component, String message) {
        if (level < currentLevel) return;
        if (!initialized) init();  // Auto-init if needed
        
        String levelStr;
        switch (level) {
            case DEBUG: levelStr = "DEBUG"; break;
            case INFO:  levelStr = "INFO "; break;
            case WARN:  levelStr = "WARN "; break;
            case ERROR: levelStr = "ERROR"; break;
            default:    levelStr = "?????"; break;
        }
        
        String wallClock = timeFormat.format(new Date());
        double simTime = getSimulationTime();
        String simTimeStr = simTime >= 0 ? String.format("t=%8.2f", simTime) : "t=    N/A";
        
        String logLine = String.format("[%s] [%s] [%s] [%-15s] %s", 
                                       wallClock, simTimeStr, levelStr, component, message);
        
        // Console output
        if (level >= WARN) {
            System.err.println(logLine);
        } else {
            System.out.println(logLine);
        }
        
        // File output
        if (fileWriter != null) {
            fileWriter.println(logLine);
            fileWriter.flush();
        }
    }
    
    /**
     * Get current simulation time from CloudSim.
     */
    private static double getSimulationTime() {
        try {
            Class<?> cloudSimClass = Class.forName("org.cloudbus.cloudsim.core.CloudSim");
            java.lang.reflect.Method clockMethod = cloudSimClass.getMethod("clock");
            return (Double) clockMethod.invoke(null);
        } catch (Exception e) {
            return -1;
        }
    }

    private static String repeatChar(char c, int count) {
        StringBuilder sb = new StringBuilder(count);
        for (int i = 0; i < count; i++) {
            sb.append(c);
        }
        return sb.toString();
    }
    
    // ==================== Special Logging Methods ====================
    
    /**
     * Log a section separator for readability.
     */
    public static void section(String title) {
        String line = repeatChar('=', 70);
        log(INFO, "SECTION", line);
        log(INFO, "SECTION", "  " + title);
        log(INFO, "SECTION", line);
    }
    
    /**
     * Log JSON data (truncated if too long, full in file).
     */
    public static void logJson(String component, String label, String json) {
        String display = json;
        boolean truncated = false;
        if (json.length() > 300) {
            display = json.substring(0, 300) + "...";
            truncated = true;
        }
        
        debug(component, label + ": " + display);
        
        // Write full JSON to file only
        if (fileWriter != null && truncated) {
            fileWriter.println("    [FULL JSON " + json.length() + " chars]: " + json);
            fileWriter.flush();
        }
    }

    
        
    /**
     * Log a formatted table header.
     */
    public static void tableHeader(String component, String... columns) {
        StringBuilder sb = new StringBuilder("| ");
        StringBuilder sep = new StringBuilder("+-");
        for (String col : columns) {
            sb.append(String.format("%-12s | ", col));
            sep.append(repeatChar('-', 12)).append("-+-");
        }
        debug(component, sep.toString());
        debug(component, sb.toString());
        debug(component, sep.toString());
    }
    
    /**
     * Log a table row.
     */
    public static void tableRow(String component, Object... values) {
        StringBuilder sb = new StringBuilder("| ");
        for (Object val : values) {
            sb.append(String.format("%-12s | ", val.toString()));
        }
        debug(component, sb.toString());
    }
    
    // ==================== Lifecycle ====================
    
    /**
     * Close the logger. Call at shutdown.
     */
    public static synchronized void close() {
        if (fileWriter != null) {
            fileWriter.println();
            fileWriter.println("================================================================================");
            fileWriter.println("LOG ENDED: " + dateFormat.format(new Date()));
            fileWriter.println("================================================================================");
            fileWriter.close();
            fileWriter = null;
            
            System.out.println("[CFR-RL] ========================================");
            System.out.println("[CFR-RL] Log file closed: " + new File(logFilePath).getAbsolutePath());
            System.out.println("[CFR-RL] ========================================");
        }
        initialized = false;
    }
    
    /**
     * Get the log file path.
     */
    public static String getLogFilePath() {
        return new File(logFilePath).getAbsolutePath();
    }
    
    /**
     * Check if logger is initialized.
     */
    public static boolean isInitialized() {
        return initialized;
    }
}