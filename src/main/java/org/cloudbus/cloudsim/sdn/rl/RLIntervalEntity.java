package org.cloudbus.cloudsim.sdn.rl;

import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.core.SimEntity;
import org.cloudbus.cloudsim.core.SimEvent;
import org.cloudbus.cloudsim.sdn.rl.RLPipe;

import java.util.HashMap;
import java.util.Map;

public class RLIntervalEntity extends SimEntity {

    private static final int RL_EVENT = 20001;   // custom event ID
    private double interval;
    private RLPipe pipe;
    private int counter = 0;

    public RLIntervalEntity(String name, double interval, RLPipe pipe) {
        super(name);
        this.interval = interval;
        this.pipe = pipe;
    }

    @Override
    public void startEntity() {
        // Schedule first RL event
        schedule(getId(), interval, RL_EVENT);
    }

    @Override
    public void processEvent(SimEvent ev) {
        if (ev.getTag() == RL_EVENT) {
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
