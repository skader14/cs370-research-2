/*
 * Title:        CloudSimSDN
 * Description:  SDN extension for CloudSim
 * Licence:      GPL - http://www.gnu.org/copyleft/gpl.html
 *
 * Copyright (c) 2015, The University of Melbourne, Australia
 */

package org.cloudbus.cloudsim.sdn;

import org.cloudbus.cloudsim.sdn.workload.Request;

/**
 * Network data packet to transfer from source to destination.
 * Payload of Packet will have a list of activities. 
 *  
 * @author Jungmin Son
 * @author Rodrigo N. Calheiros
 * @since CloudSimSDN 1.0
 */
public class Packet {
	private static long automaticPacketId = 0;
	private final long id;
	private int origin;			// origin VM adress (vm.getId())
	private int destination;	// destination VM adress (vm.getId())
	private final long size;
	private final int flowId;
	private Request payload;

	private double startTime=-1;
	private double finishTime=-1;

	private Packet pktEncapsulated = null;
	
	public Packet(int origin, int destination, long size, int flowId, Request payload) {
		this.origin = origin;
		this.destination = destination;
		this.size = size;
		this.flowId = flowId;
		this.payload = payload;
		this.id = automaticPacketId++;
		
		if(size < 0) {
			throw new RuntimeException("Packet size cannot be minus! Pkt="+this+", size="+size);
		}
	}
	
	public Packet(int origin, int destination, long size, int flowId, Request payload, Packet encapsulatedPkt) { 
		this(origin, destination, size, flowId, payload);
		this.pktEncapsulated = encapsulatedPkt; 
	}
	
	public int getOrigin() {
		return origin;
	}
	
	public void changeOrigin(int vmId) {
		origin = vmId;
	}

	public int getDestination() {
		return destination;
	}

	public void changeDestination(int vmId) {
		destination = vmId;
	}
	
	public long getSize() {
		return size;
	}

	public Request getPayload() {
		return payload;
	}
	
	public int getFlowId() {
		return flowId;
	}
	
	public String toString() {
		return "PKG:"+origin + "->" + destination + " - " + payload.toString();
	}
	
	@Override
	public int hashCode() {
		return Long.hashCode(id);
	}

	public void setPacketStartTime(double time) {
		this.startTime = time;
		
		if(pktEncapsulated != null && pktEncapsulated.getStartTime() == -1) {
			pktEncapsulated.setPacketStartTime(time);
		}
	}
	
	public void setPacketFinishTime(double time) {
		this.finishTime = time;
		
		if(pktEncapsulated != null) {
			pktEncapsulated.setPacketFinishTime(time);
		}
	}
	
	public void setPacketFailedTime(double currentTime) {
		setPacketFinishTime(currentTime);
		getPayload().setFailedTime(currentTime);
		if(pktEncapsulated != null) {
			pktEncapsulated.setPacketFailedTime(currentTime);
		}
	}
	
	public double getStartTime() {
		//if(pktEncapsulated != null) {
		//	return pktEncapsulated.getStartTime();
		//}
		
		return this.startTime;
	}
	
	public double getFinishTime() {
		//if(pktEncapsulated != null) {
		//	return pktEncapsulated.getFinishTime();
		//}
		
		return this.finishTime;
	}
	
	public long getPacketId() {
		return this.id;
	}

	// ============================================================================
	// ADD THESE TO Packet.java (org.cloudbus.cloudsim.sdn.Packet)
	// Add fields near other private fields, and methods at end of class
	// ============================================================================

	// ============ Channel Info for Latency Recording ============
	// These are set in NetworkOperatingSystem.processCompletePackets() 
	// BEFORE the channel is removed from channelTable.
	// This solves the "findChannel returns null" bug.

	private double channelTotalLatency = 0;      // Propagation delay (seconds)
	private double channelBandwidth = 0;          // Allocated bandwidth (bps)
	private int channelPathLength = 0;            // Number of hops

	/**
	 * Store channel info on this packet before the channel is removed.
	 * Called by NetworkOperatingSystem.processCompletePackets()
	 * 
	 * @param totalLatency  Channel's propagation delay (from getTotalLatency())
	 * @param bandwidth     Channel's allocated bandwidth (from getAllocatedBandwidth())
	 * @param pathLength    Number of hops in path (from getPathLength())
	 */
	public void setChannelInfo(double totalLatency, double bandwidth, int pathLength) {
		this.channelTotalLatency = totalLatency;
		this.channelBandwidth = bandwidth;
		this.channelPathLength = pathLength;
	}

	/**
	 * Get the propagation delay that was stored before channel was removed.
	 * @return Propagation delay in seconds, or 0 if not set
	 */
	public double getChannelTotalLatency() {
		return channelTotalLatency;
	}

	/**
	 * Get the bandwidth that was stored before channel was removed.
	 * @return Bandwidth in bps, or 0 if not set
	 */
	public double getChannelBandwidth() {
		return channelBandwidth;
	}

	/**
	 * Get the path length that was stored before channel was removed.
	 * @return Number of hops, or 0 if not set
	 */
	public int getChannelPathLength() {
		return channelPathLength;
	}

}
