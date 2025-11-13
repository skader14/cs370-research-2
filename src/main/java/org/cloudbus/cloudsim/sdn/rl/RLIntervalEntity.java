package org.cloudbus.cloudsim.sdn.rl;

import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.core.SimEntity;
import org.cloudbus.cloudsim.core.SimEvent;
import org.cloudbus.cloudsim.sdn.rl.RLPipe;
import org.cloudbus.cloudsim.sdn.nos.NetworkOperatingSystem;
import org.cloudbus.cloudsim.sdn.nos.NetworkOperatingSystemSimple;

import java.util.HashMap;
import java.util.Map;

public class RLIntervalEntity extends SimEntity {

    private static final int RL_EVENT = 20001;   // custom event ID
    private double interval;
    private RLPipe pipe;
    private int counter = 0;
    private NetworkOperatingSystem nos;

    public RLIntervalEntity(String name, double interval, RLPipe pipe, NetworkOperatingSystem nos) {
        super(name);
        this.interval = interval;
        this.pipe = pipe;
        this.nos = nos;
    }

    @Override
    public void startEntity() {
        // Schedule first RL event
        schedule(getId(), interval, RL_EVENT);
    }

    @Override
    public void processEvent(SimEvent ev) {
        if (ev.getTag() == RL_EVENT) {
            

            // 1. Stop if CloudSim is trying to finish
            if (!CloudSim.running()) return;

            // 2. Stop if no flows exist in the network
            if (nos.getChannelManager().getTotalChannelNum() == 0) {
                System.out.println("RLIntervalEntity: No flows left, stopping RL.");
                return;
            }



            counter++;

            // Turn state into JSON string
            StringBuilder sb = new StringBuilder();
            sb.append("{");
            sb.append("\"times_called\":").append(counter).append(",");
            sb.append("\"sim_time\":").append(CloudSim.clock());
            sb.append("}");
            String stateJson = sb.toString();

            try {
                pipe.sendState(stateJson);
                String response = pipe.receiveAction();
                System.out.println("[RLIntervalEntity] Received from Python: " + response);
            } catch (Exception e) {
                e.printStackTrace();
            }

            // schedule next interval
            schedule(getId(), interval, RL_EVENT);
        }
    }

    @Override
    public void shutdownEntity() {
        System.out.println("[RLIntervalEntity] Shutting down.");
    }
}
