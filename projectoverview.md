# CFR-RL: Latency-Aware Traffic Engineering with Reinforcement Learning

Extension of the [CFR-RL framework](https://arxiv.org/abs/2004.09803) that trains an RL agent to minimize latency in data center networks using packet-level simulation feedback via CloudSimSDN.

---

## Overview

Standard traffic engineering optimizes for link utilization. This project extends CFR-RL to optimize for **queuing latency** — what applications actually experience. Instead of rerouting all flows, the policy learns to identify a small subset of "critical flows" where smart rerouting has the most impact, leaving everything else on ECMP.

**Stack:** Python (PyTorch) + Java (CloudSimSDN) + Maven

---

## Requirements

- Java 8+, Maven
- Python 3.8+

```bash
pip install torch numpy pandas
mvn compile
```

---

## Training

```bash
python RL/csimtraining/cloudsim_trainer.py \
    --cloudsim-dir . \
    --episodes 500 \
    --algorithm ppo \
    --dense-rewards \
    --packets 2000 \
    --traffic-model hotspot \
    --output-dir training_outputs
```

Resume from checkpoint:
```bash
python RL/csimtraining/cloudsim_trainer.py \
    --cloudsim-dir . \
    --resume training_outputs/checkpoints/checkpoint_ep250.pt \
    --episodes 250
```

---

## Project Structure

```
cloudsimsdn/
├── src/org/cloudbus/cloudsim/sdn/
│   ├── example/CFRRLTrainingRunner.java   # Java simulation entry point
│   └── rl/
│       ├── LinkStatsCollector.java
│       └── LatencyCollector.java
├── RL/csimtraining/
│   ├── cloudsim_trainer.py                # Main training loop + CLI
│   ├── policy_network.py                  # MLP policy (~790K params)
│   ├── ppo_trainer.py                     # PPO implementation
│   ├── workload_generator.py              # Traffic generation
│   ├── feature_extractor.py               # 9 per-flow features
│   └── episode_runner.py                  # Java subprocess manager
└── dataset-fattree/                       # Fat-Tree k=4 topology
```

---

## Results

Trained on Fat-Tree k=4, 500 episodes (PPO + dense rewards):

| Baseline | vs. Trained Policy |
|----------|--------------------|
| ECMP (no rerouting) | 13.5% worse (p=0.0035) |
| Top-K by demand | 27.3% worse (p<0.0001) |
| Top-K by queuing delay | ~same (p=0.94) |

Learning curve: **21.7% improvement** from first to last 100 episodes (p=0.0039).

---

## Key Implementation Notes

**Java–Python bridge:** Each episode spawns CloudSimSDN as a subprocess. Python writes the selected critical flows to a file, Java runs the simulation and outputs results to JSON, Python reads and computes reward. ~5–10s per episode.

**Race condition fix:** CloudSimSDN's default packet completion flow deleted channel objects before latency could be recorded. Fixed by storing channel metadata on the `Packet` object in `processCompletePackets()` before `updateChannel()` removes it.

**Topology matters:** Abilene (12-node backbone) failed to produce a learning signal — too few alternative paths for routing decisions to differentiate. Fat-Tree k=4 provides 4 paths per cross-pod flow and is the minimum viable topology for this approach.

**Algorithm progression:** REINFORCE (no learning, p=0.23) → PPO sparse (learning detected, p=0.014) → PPO dense (strong learning, p=0.0039). Dense rewards provide ~14x more gradient updates per episode via per-second windowed feedback.

---

## References

- Zhang et al., [CFR-RL: Traffic Engineering With Reinforcement Learning in SDN](https://arxiv.org/abs/2004.09803)
- Al-Fares et al., A Scalable, Commodity Data Center Network Architecture (Fat-Tree)
- [CloudSimSDN](https://github.com/Cloudslab/CloudSimSDN)
- Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)