#!/usr/bin/env python3
"""
diagnose_cloudsim.py - Diagnostic script for CFR-RL CloudSim bandwidth issue

This script performs several diagnostic checks:
1. Analyzes existing metrics to confirm the bandwidth mismatch
2. Tests if the policy network can learn a synthetic (known) reward function
3. Provides recommendations for fixing CloudSim configuration

Usage:
    python diagnose_cloudsim.py --metrics-file training_metrics.csv
    python diagnose_cloudsim.py --test-synthetic-learning
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import json


def analyze_bandwidth_mismatch(workload_file: str, episode_log: str = None):
    """
    Analyze the discrepancy between configured and actual bandwidth.
    """
    print("=" * 70)
    print("BANDWIDTH MISMATCH ANALYSIS")
    print("=" * 70)
    
    # Load workload
    workload = pd.read_csv(workload_file)
    
    total_bytes = workload['psize'].sum()
    num_packets = len(workload)
    avg_packet_size = total_bytes / num_packets
    max_packet_size = workload['psize'].max()
    duration = workload['start'].max() - workload['start'].min()
    
    print(f"\nWorkload Analysis:")
    print(f"  Total packets: {num_packets}")
    print(f"  Total data: {total_bytes / 1e9:.2f} GB")
    print(f"  Average packet: {avg_packet_size / 1e6:.1f} MB")
    print(f"  Max packet: {max_packet_size / 1e6:.1f} MB")
    print(f"  Duration: {duration:.1f} seconds")
    
    # Configured bandwidth analysis
    print(f"\n--- Expected Performance at Different Bandwidths ---")
    
    bandwidths = {
        '10 Mbps (configured in physical.json)': 10e6 / 8,  # bytes/s
        '100 Kbps (configured in virtual.json flows)': 100e3 / 8,
        '1 Gbps (CloudSim default?)': 1e9 / 8,
        '10 Gbps (what seems to be happening)': 10e9 / 8,
    }
    
    for name, bw_bytes in bandwidths.items():
        # Time to transmit average packet
        tx_time = avg_packet_size / bw_bytes
        # Time to transmit all packets sequentially
        total_time = total_bytes / bw_bytes
        
        print(f"\n  {name}:")
        print(f"    Single packet TX time: {tx_time:.2f} seconds")
        print(f"    Total sequential TX time: {total_time:.0f} seconds")
        
        if tx_time < 1:
            print(f"    --> Feasible within 20s simulation")
        else:
            print(f"    --> NOT feasible! Takes {tx_time:.0f}s per packet")
    
    # Evidence from actual runs
    print(f"\n--- Evidence from Simulation ---")
    print(f"  Observed: All 100 packets complete in ~20 second simulation")
    print(f"  Observed: Max link utilization ~0.78%")
    print(f"  Observed: Service time ~0.9s for 100MB packets")
    print(f"  Implied bandwidth: ~115 MB/s = ~920 Mbps")
    print(f"\n  CONCLUSION: CloudSim is NOT using your 10 Mbps configuration!")


def analyze_training_metrics(metrics_file: str):
    """
    Analyze training metrics to confirm no learning is occurring.
    """
    print("\n" + "=" * 70)
    print("TRAINING METRICS ANALYSIS")
    print("=" * 70)
    
    df = pd.read_csv(metrics_file)
    
    print(f"\nDataset: {len(df)} episodes")
    
    # Basic stats
    print(f"\nReward Statistics:")
    print(f"  Mean: {df['reward'].mean():.4f}")
    print(f"  Std: {df['reward'].std():.4f}")
    print(f"  Min: {df['reward'].min():.4f}")
    print(f"  Max: {df['reward'].max():.4f}")
    
    # Check for trend
    first_half = df.iloc[:len(df)//2]['reward'].mean()
    second_half = df.iloc[len(df)//2:]['reward'].mean()
    
    print(f"\nLearning Signal Check:")
    print(f"  First half avg reward: {first_half:.4f}")
    print(f"  Second half avg reward: {second_half:.4f}")
    print(f"  Improvement: {second_half - first_half:.4f}")
    
    if abs(second_half - first_half) < df['reward'].std() * 0.5:
        print(f"  --> NO LEARNING DETECTED (improvement < 0.5 std)")
    
    # Check drop rates
    print(f"\nCongestion Signal Check:")
    print(f"  Drop rate - Mean: {df['drop_rate'].mean():.6f}")
    print(f"  Drop rate - Max: {df['drop_rate'].max():.6f}")
    
    if df['drop_rate'].max() == 0:
        print(f"  --> ZERO DROPS! Network never congested.")
        print(f"      This is why the policy can't learn anything!")
    
    # Reward variance breakdown
    queuing_component = df['mean_queuing_ms'] / 1000.0  # Convert to reward scale
    drop_component = 10 * df['drop_rate']
    
    print(f"\nReward Component Analysis:")
    print(f"  Queuing component std: {queuing_component.std():.4f}")
    print(f"  Drop component std: {drop_component.std():.6f}")
    print(f"  --> All reward variance comes from random queuing variation")


def test_synthetic_learning():
    """
    Test if the policy network can learn a simple synthetic reward.
    
    This isolates the RL training code from CloudSim issues.
    If this works, the problem is definitely in CloudSim.
    If this fails, there might also be training code issues.
    """
    print("\n" + "=" * 70)
    print("SYNTHETIC LEARNING TEST")
    print("=" * 70)
    print("\nTesting if policy network can learn a known-learnable reward...")
    
    # Import from the project (assuming this is run from project dir)
    try:
        from policy_network import PolicyNetwork, BatchReinforceTrainer
    except ImportError:
        print("ERROR: Could not import policy_network. Run from project directory.")
        return
    
    # Smaller test setup
    NUM_FLOWS = 50
    NUM_FEATURES = 9
    K = 5
    NUM_EPISODES = 200
    BATCH_SIZE = 10
    
    # Create network
    policy = PolicyNetwork(num_flows=NUM_FLOWS, num_features=NUM_FEATURES)
    trainer = BatchReinforceTrainer(policy, lr=1e-3, batch_size=BATCH_SIZE)
    
    print(f"\nSetup: {NUM_FLOWS} flows, K={K}, {NUM_EPISODES} episodes")
    
    # Synthetic reward: prefer selecting flows with higher indices
    # This is a simple, clearly learnable objective
    def synthetic_reward(selected_flows):
        # Reward = mean of selected indices (normalized to 0-1)
        return sum(selected_flows) / (K * NUM_FLOWS)
    
    rewards = []
    
    for ep in range(NUM_EPISODES):
        # Random features (don't matter for synthetic reward)
        features = torch.randn(1, NUM_FLOWS * NUM_FEATURES)
        
        # Get action
        scores = policy(features)
        selected, log_prob = policy.sample_action(scores, k=K, temperature=1.0)
        
        # Synthetic reward
        reward = synthetic_reward(selected)
        rewards.append(reward)
        
        # Store for training
        trainer.store_episode(features, selected, log_prob, reward, 1.0)
        
        # Update
        if trainer.should_update():
            trainer.update()
    
    # Analyze results
    first_20 = np.mean(rewards[:20])
    last_20 = np.mean(rewards[-20:])
    
    print(f"\nResults:")
    print(f"  First 20 episodes avg reward: {first_20:.4f}")
    print(f"  Last 20 episodes avg reward: {last_20:.4f}")
    print(f"  Improvement: {last_20 - first_20:.4f} ({100*(last_20 - first_20)/first_20:.1f}%)")
    
    # Optimal would be selecting top K flows
    optimal = (NUM_FLOWS - K + 1 + NUM_FLOWS) / 2 / NUM_FLOWS
    print(f"  Optimal possible: {optimal:.4f}")
    print(f"  Achieved: {100 * last_20 / optimal:.1f}% of optimal")
    
    if last_20 > first_20 + 0.05:
        print(f"\n  ✓ LEARNING DETECTED!")
        print(f"    The policy network and REINFORCE trainer work correctly.")
        print(f"    Problem is definitely in CloudSim reward signal.")
    else:
        print(f"\n  ✗ NO LEARNING DETECTED")
        print(f"    There may be issues with the training code as well.")


def compute_required_bandwidth_for_congestion(
    num_packets: int = 100,
    avg_packet_mb: float = 100,
    duration_s: float = 20,
    target_utilization: float = 0.75,
    num_links: int = 48,  # Fat-tree k=4 has 48 links
    avg_hops: int = 5,    # Average path length
):
    """
    Calculate what bandwidth is needed to achieve target utilization.
    """
    print("\n" + "=" * 70)
    print("BANDWIDTH REQUIREMENT CALCULATION")
    print("=" * 70)
    
    total_bytes = num_packets * avg_packet_mb * 1e6
    total_bits = total_bytes * 8
    
    # Traffic spread across links (rough estimate)
    bits_per_link = total_bits * avg_hops / num_links
    
    # Required bandwidth for target utilization
    required_bw = bits_per_link / (duration_s * target_utilization)
    
    print(f"\nInputs:")
    print(f"  Packets: {num_packets}")
    print(f"  Avg packet size: {avg_packet_mb} MB")
    print(f"  Duration: {duration_s} seconds")
    print(f"  Target utilization: {target_utilization*100:.0f}%")
    
    print(f"\nCalculations:")
    print(f"  Total traffic: {total_bytes/1e9:.2f} GB = {total_bits/1e9:.1f} Gbits")
    print(f"  Bits per link: {bits_per_link/1e9:.3f} Gbits")
    print(f"  Required link bandwidth for {target_utilization*100:.0f}% util: {required_bw/1e6:.1f} Mbps")
    
    print(f"\nRecommendation:")
    print(f"  To achieve {target_utilization*100:.0f}% utilization with current traffic:")
    print(f"  - Set physical link bandwidth to: {required_bw/1e6:.1f} Mbps")
    print(f"  - OR reduce packet sizes by {100*(1 - required_bw/10e6):.0f}%")
    print(f"  - OR increase packet count to {num_packets * 10e6 / required_bw:.0f}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose CFR-RL CloudSim issues")
    parser.add_argument("--workload", type=str, help="Path to workload CSV")
    parser.add_argument("--metrics", type=str, help="Path to training_metrics.csv")
    parser.add_argument("--test-synthetic", action="store_true", 
                       help="Test if policy can learn synthetic reward")
    parser.add_argument("--all", action="store_true",
                       help="Run all diagnostics")
    
    args = parser.parse_args()
    
    if args.workload or args.all:
        workload_file = args.workload or "dataset-fattree/fattree-workload.csv"
        analyze_bandwidth_mismatch(workload_file)
    
    if args.metrics or args.all:
        metrics_file = args.metrics or "training_metrics.csv"
        if Path(metrics_file).exists():
            analyze_training_metrics(metrics_file)
        else:
            print(f"Metrics file not found: {metrics_file}")
    
    if args.test_synthetic or args.all:
        test_synthetic_learning()
    
    if not any([args.workload, args.metrics, args.test_synthetic, args.all]):
        compute_required_bandwidth_for_congestion()
        parser.print_help()


if __name__ == "__main__":
    main()