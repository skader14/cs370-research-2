package org.cloudbus.cloudsim.sdn.rl;

import java.io.*;
import java.nio.charset.StandardCharsets;

public class RLPipe {

    private Process pythonProcess;
    private BufferedWriter pythonWriter;
    private BufferedReader pythonReader;
    private BufferedReader pythonErrorReader;  // Separate stderr reader
    private Thread stderrThread;  // Thread to consume stderr
    private boolean isRunning = false;

    private final String pythonCommand;
    private final String scriptPath;   // path relative to project root

    public RLPipe(String pythonCommand) {
        this.pythonCommand = pythonCommand;

        // Python file lives in ROOT/RL/agent.py
        this.scriptPath = "RL/agent.py";
    }

    public void startPython() throws IOException {
        if (isRunning) return;

        ProcessBuilder pb = new ProcessBuilder(
                pythonCommand,
                scriptPath
        );

        // DO NOT merge stderr into stdout - keep them separate!
        pb.redirectErrorStream(false);
        pb.directory(new File("."));   // ensure root directory as working dir

        pythonProcess = pb.start();
        pythonWriter = new BufferedWriter(new OutputStreamWriter(
                pythonProcess.getOutputStream(), StandardCharsets.UTF_8));
        pythonReader = new BufferedReader(new InputStreamReader(
                pythonProcess.getInputStream(), StandardCharsets.UTF_8));
        pythonErrorReader = new BufferedReader(new InputStreamReader(
                pythonProcess.getErrorStream(), StandardCharsets.UTF_8));

        // Start a thread to consume stderr and print it (prevents blocking)
        stderrThread = new Thread(() -> {
            try {
                String line;
                while ((line = pythonErrorReader.readLine()) != null) {
                    // Print Python's log messages to Java's stderr
                    System.err.println("[Python] " + line);
                }
            } catch (IOException e) {
                // Process ended, ignore
            }
        }, "Python-stderr-reader");
        stderrThread.setDaemon(true);
        stderrThread.start();

        isRunning = true;
        System.out.println("[RLPipe] Started Python: " + pythonCommand + " " + scriptPath);
    }

    public void sendState(String json) throws IOException {
        pythonWriter.write(json);
        pythonWriter.write("\n");
        pythonWriter.flush();
    }

    public String receiveAction() throws IOException {
        String line = pythonReader.readLine();
        if (line == null)
            throw new IOException("Python process died.");
        return line;
    }

    public void close() {
        try {
            if (pythonWriter != null) {
                pythonWriter.write("{\"shutdown\": true}\n");
                pythonWriter.flush();
            }
        } catch (IOException ignored) {}

        if (pythonProcess != null) pythonProcess.destroy();

        System.out.println("[RLPipe] Closed Python process.");
    }
}