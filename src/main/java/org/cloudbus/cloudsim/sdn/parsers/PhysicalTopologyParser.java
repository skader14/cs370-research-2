/*
 * PhysicalTopologyParser.java - FIXED VERSION
 * 
 * KEY FIX: The original parseLink() method ignored bandwidth from the JSON file
 * and hardcoded -1. This version properly reads the 'bw' field from links OR
 * inherits it from nodes.
 * 
 * CHANGES MADE:
 * 1. parseLink() now reads 'bw' field from each link in JSON
 * 2. If link doesn't have 'bw', it inherits from the source node
 * 3. Added debug logging to verify bandwidth is being applied
 */

package org.cloudbus.cloudsim.sdn.parsers;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.cloudbus.cloudsim.sdn.HostFactory;
import org.cloudbus.cloudsim.sdn.HostFactorySimple;
import org.cloudbus.cloudsim.sdn.nos.NetworkOperatingSystem;
import org.cloudbus.cloudsim.sdn.nos.NetworkOperatingSystemSimple;
import org.cloudbus.cloudsim.sdn.physicalcomponents.Link;
import org.cloudbus.cloudsim.sdn.physicalcomponents.Node;
import org.cloudbus.cloudsim.sdn.physicalcomponents.SDNHost;
import org.cloudbus.cloudsim.sdn.physicalcomponents.switches.AggregationSwitch;
import org.cloudbus.cloudsim.sdn.physicalcomponents.switches.CoreSwitch;
import org.cloudbus.cloudsim.sdn.physicalcomponents.switches.EdgeSwitch;
import org.cloudbus.cloudsim.sdn.physicalcomponents.switches.GatewaySwitch;
import org.cloudbus.cloudsim.sdn.physicalcomponents.switches.IntercloudSwitch;
import org.cloudbus.cloudsim.sdn.physicalcomponents.switches.Switch;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.JSONValue;

import org.cloudbus.cloudsim.sdn.rl.CFRRLLogger;

import com.google.common.collect.HashMultimap;
import com.google.common.collect.Multimap;

/**
 * This class parses Physical Topology JSON file.
 * It supports multiple data centers.
 * 
 * FIXED: Now properly reads bandwidth from topology JSON.
 * 
 * @author Jungmin Son
 * @since CloudSimSDN 1.0
 */
public class PhysicalTopologyParser {
    private String filename;

    private Multimap<String, SDNHost> sdnHosts;
    private Multimap<String, Switch> switches;
    private List<Link> links = new ArrayList<Link>();
    private Hashtable<String, Node> nameNodeTable = new Hashtable<String, Node>();
    
    // ADDED: Store node bandwidths for inheritance
    private Hashtable<String, Long> nodeBandwidths = new Hashtable<String, Long>();
    
    // ADDED: Default bandwidth if not specified anywhere
    private static final long DEFAULT_BANDWIDTH = 1_000_000_000L; // 1 Gbps default
    
    private HostFactory hostFactory = null;
    
    public PhysicalTopologyParser(String jsonFilename, HostFactory hostFactory) {
        sdnHosts = HashMultimap.create();
        switches = HashMultimap.create();
        this.hostFactory = hostFactory;
        
        this.filename = jsonFilename;
    }

    public static Map<String, NetworkOperatingSystem> loadPhysicalTopologyMultiDC(String physicalTopologyFilename) {
        PhysicalTopologyParser parser = new PhysicalTopologyParser(physicalTopologyFilename, new HostFactorySimple());
        Map<String, String> dcNameType = parser.parseDatacenters();
        Map<String, NetworkOperatingSystem> netOsList = new HashMap<String, NetworkOperatingSystem>();
        
        for(String dcName: dcNameType.keySet()) {
            NetworkOperatingSystem nos;
            nos = new NetworkOperatingSystemSimple("NOS_"+dcName);
            
            netOsList.put(dcName, nos);
            parser.parseNode(dcName);
        }
        parser.parseLink();
        
        for(String dcName: dcNameType.keySet()) {
            if(!"network".equals(dcNameType.get(dcName))) {
                NetworkOperatingSystem nos = netOsList.get(dcName);
                nos.configurePhysicalTopology(parser.getHosts(dcName), parser.getSwitches(dcName), parser.getLinks());
            }
        }
        for(String dcName: dcNameType.keySet()) {
            if("network".equals(dcNameType.get(dcName))) {
                NetworkOperatingSystem nos = netOsList.get(dcName);
                nos.configurePhysicalTopology(parser.getHosts(dcName), parser.getSwitches(dcName), parser.getLinks());
            }
        }

        return netOsList;
    }
    
    public static void loadPhysicalTopologySingleDC(String physicalTopologyFilename, NetworkOperatingSystem nos, HostFactory hostFactory) {
        PhysicalTopologyParser parser = new PhysicalTopologyParser(physicalTopologyFilename, hostFactory);
        parser.parse(nos);
        nos.configurePhysicalTopology(parser.getHosts(), parser.getSwitches(), parser.getLinks());
    }
    
    public Collection<SDNHost> getHosts() {
        return this.sdnHosts.values();
    }
    
    public Collection<SDNHost> getHosts(String dcName) {
        return this.sdnHosts.get(dcName);
    }
    
    public Collection<Switch> getSwitches() {
        return this.switches.values();
    }
    
    public Collection<Switch> getSwitches(String dcName) {
        return this.switches.get(dcName);
    }
    
    public List<Link> getLinks() {
        return this.links;
    }
    
    public Map<String, String> parseDatacenters() {
        HashMap<String, String> dcNameType = new HashMap<String, String>();
        try {
            JSONObject doc = (JSONObject) JSONValue.parse(new FileReader(this.filename));
            
            JSONArray datacenters = (JSONArray) doc.get("datacenters");
            @SuppressWarnings("unchecked")
            Iterator<JSONObject> iter = datacenters.iterator(); 
            while(iter.hasNext()){
                JSONObject node = iter.next();
                String dcName = (String) node.get("name");
                String type = (String) node.get("type");
                
                dcNameType.put(dcName, type);
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        
        return dcNameType;        
    }
    
    private void parse(NetworkOperatingSystem nos) {
        parseNode(null);
        parseLink();
    }
    
    public void parseNode(String datacenterName) {
        try {
            JSONObject doc = (JSONObject) JSONValue.parse(new FileReader(this.filename));
            
            JSONArray nodes = (JSONArray) doc.get("nodes");
            @SuppressWarnings("unchecked")
            Iterator<JSONObject> iter = nodes.iterator(); 
            while(iter.hasNext()){
                JSONObject node = iter.next();
                String nodeType = (String) node.get("type");
                String nodeName = (String) node.get("name");
                String dcName = (String) node.get("datacenter");
                if(datacenterName != null && !datacenterName.equals(dcName)) {
                    continue;
                }
                
                if(nodeType.equalsIgnoreCase("host")){
                    long pes = (Long) node.get("pes");
                    long mips = (Long) node.get("mips");
                    int ram = new BigDecimal((Long)node.get("ram")).intValueExact();
                    long storage = (Long) node.get("storage");
                    long bw = new BigDecimal((Long)node.get("bw")).intValueExact();
                    
                    // ADDED: Store bandwidth for this node
                    nodeBandwidths.put(nodeName, bw);
                    System.out.println("[TOPO_DEBUG] Host " + nodeName + " bandwidth: " + bw + " bps");
					CFRRLLogger.info("TopoParser", "Host " + nodeName + " bandwidth: " + bw + " bps");
					// CFRRLLogger.info("TopoParser", "Link " + src + "->" + dst + " bw: " + bw + " bps");
                    
                    int num = 1;
                    if (node.get("nums")!= null)
                        num = new BigDecimal((Long)node.get("nums")).intValueExact();

                    for(int n = 0; n< num; n++) {
                        String nodeName2 = nodeName;
                        if(num >1) nodeName2 = nodeName + n;
                        
                        SDNHost sdnHost = hostFactory.createHost(ram, bw, storage, pes, mips, nodeName);
                        nameNodeTable.put(nodeName2, sdnHost);
                        
                        this.sdnHosts.put(dcName, sdnHost);
                    }
                    
                } else {
                    int MAX_PORTS = 256;
                            
                    long bw = new BigDecimal((Long)node.get("bw")).longValueExact();
                    long iops = (Long) node.get("iops");
                    
                    // ADDED: Store bandwidth for this switch
                    nodeBandwidths.put(nodeName, bw);
                    System.out.println("[TOPO_DEBUG] Switch " + nodeName + " (" + nodeType + ") bandwidth: " + bw + " bps");
					CFRRLLogger.info("TopoParser", "Switch " + nodeName + " (" + nodeType + ") bandwidth: " + bw + " bps");
                    
                    int upports = MAX_PORTS;
                    int downports = MAX_PORTS;
                    if (node.get("upports")!= null)
                        upports = new BigDecimal((Long)node.get("upports")).intValueExact();
                    if (node.get("downports")!= null)
                        downports = new BigDecimal((Long)node.get("downports")).intValueExact();
                    Switch sw = null;
                    
                    if(nodeType.equalsIgnoreCase("core")) {
                        sw = new CoreSwitch(nodeName, bw, iops, upports, downports);
                    } else if (nodeType.equalsIgnoreCase("aggregate")){
                        sw = new AggregationSwitch(nodeName, bw, iops, upports, downports);
                    } else if (nodeType.equalsIgnoreCase("edge")){
                        sw = new EdgeSwitch(nodeName, bw, iops, upports, downports);
                    } else if (nodeType.equalsIgnoreCase("intercloud")){
                        sw = new IntercloudSwitch(nodeName, bw, iops, upports, downports);
                    } else if (nodeType.equalsIgnoreCase("gateway")){
                        if(nameNodeTable.get(nodeName) != null)
                            sw = (Switch)nameNodeTable.get(nodeName);
                        else
                            sw = new GatewaySwitch(nodeName, bw, iops, upports, downports);
                    } else {
                        throw new IllegalArgumentException("No switch found!");
                    }
                    
                    if(sw != null) {
                        nameNodeTable.put(nodeName, sw);
                        this.switches.put(dcName, sw);
                    }
                }
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }
        
    /**
     * FIXED: parseLink now properly reads bandwidth from JSON
     * 
     * Priority for bandwidth:
     * 1. Link-specific 'bw' field in the link definition
     * 2. Source node's bandwidth (from node definition)
     * 3. Default bandwidth (1 Gbps)
     */
    public void parseLink() {
        try {
            JSONObject doc = (JSONObject) JSONValue.parse(new FileReader(this.filename));
            
            JSONArray links = (JSONArray) doc.get("links");
            @SuppressWarnings("unchecked")
            Iterator<JSONObject> linksIter = links.iterator(); 
            while(linksIter.hasNext()){
                JSONObject link = linksIter.next();
                String src = (String) link.get("source");  
                String dst = (String) link.get("destination");
                double lat = (Double) link.get("latency");
                
                // FIXED: Read bandwidth from link, or inherit from source node
                long bw;
                if (link.get("bw") != null) {
                    // Link has explicit bandwidth
                    bw = new BigDecimal((Long) link.get("bw")).longValueExact();
                    System.out.println("[TOPO_DEBUG] Link " + src + "->" + dst + " explicit bw: " + bw + " bps");
					CFRRLLogger.info("TopoParser", "Link " + src + "->" + dst + " bw: " + bw + " bps");
                } else if (nodeBandwidths.containsKey(src)) {
                    // Inherit from source node
                    bw = nodeBandwidths.get(src);
                    System.out.println("[TOPO_DEBUG] Link " + src + "->" + dst + " inherited bw from " + src + ": " + bw + " bps");
					CFRRLLogger.info("TopoParser", "Link " + src + "->" + dst + " bw: " + bw + " bps");
                } else if (nodeBandwidths.containsKey(dst)) {
                    // Inherit from destination node
                    bw = nodeBandwidths.get(dst);
                    System.out.println("[TOPO_DEBUG] Link " + src + "->" + dst + " inherited bw from " + dst + ": " + bw + " bps");
					CFRRLLogger.info("TopoParser", "Link " + src + "->" + dst + " bw: " + bw + " bps");
                } else {
                    // Use default
                    bw = DEFAULT_BANDWIDTH;
                    System.out.println("[TOPO_DEBUG] Link " + src + "->" + dst + " using DEFAULT bw: " + bw + " bps");
					CFRRLLogger.info("TopoParser", "Link " + src + "->" + dst + " bw: " + bw + " bps");
                }
                
                Node srcNode = nameNodeTable.get(src);
                Node dstNode = nameNodeTable.get(dst);
                
                // FIXED: Pass actual bandwidth instead of -1
                Link l = new Link(srcNode, dstNode, lat, bw);
                this.links.add(l);
            }
            
            System.out.println("[TOPO_DEBUG] Total links created: " + this.links.size());
			CFRRLLogger.info("TopoParser", "Total links created: " + this.links.size());
            
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }
    
    public Hashtable<String, Node> getNameNode() {
        return nameNodeTable;
    }
}