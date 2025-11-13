package org.cloudbus.cloudsim.sdn.example;

import org.cloudbus.cloudsim.sdn.rl.RLPipe;

public class RLTestPipeRunner {

    public static void main(String[] args) {
        System.out.println("[TEST] Starting RLTestPipeRunner...");

        RLPipe pipe = new RLPipe("python");   // uses RL/agent.py automatically

        try {
            pipe.startPython();

            // We will send 5 test messages
            for (int i = 1; i <= 5; i++) {
                String state = "{\"test_step\":" + i + "}";

                System.out.println("[TEST] Java sending: " + state);
                pipe.sendState(state);

                // BLOCKS until Python replies
                String action = pipe.receiveAction();
                System.out.println("[TEST] Java received: " + action);

                // Just a short sleep to make logs readable
                Thread.sleep(300);
            }

        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            System.out.println("[TEST] Sending shutdown to Python...");
            pipe.close();
            System.out.println("[TEST] RLTestPipeRunner finished.");
        }
    }
}
