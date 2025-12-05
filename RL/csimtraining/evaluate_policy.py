"""
evaluate_policy.py - Compare trained policy against random baseline.

Usage:
    python evaluate_policy.py --checkpoint training_outputs/checkpoints/best_model.pt --episodes 20
"""

import argparse
import torch
import numpy as np
from pathlib import Path

from workload_generator import generate_workload, save_workload, generate_demand_vector
from feature_extractor import FeatureExtractor, NUM_FLOWS
from policy_network import PolicyNetwork
from episode_runner import EpisodeRunner, compute_reward


def evaluate_policy(
    checkpoint_path: str,
    cloudsim_dir: str,
    num_episodes: int = 20,
    packets: int = 300,
    output_dir: str = "eval_outputs",
):
    """Evaluate trained policy vs random baseline."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load trained policy
    print(f"Loading checkpoint: {checkpoint_path}")
    policy = PolicyNetwork().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()
    
    # Initialize components
    extractor = FeatureExtractor(random_cold_start=True, seed=42)
    runner = EpisodeRunner(cloudsim_dir=cloudsim_dir, verbose=False)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Results storage
    trained_results = []
    random_results = []
    
    print(f"\nEvaluating over {num_episodes} episodes...")
    print("=" * 60)
    
    for ep in range(num_episodes):
        # Generate same workload for both policies
        seed = 1000 + ep  # Fixed seeds for fair comparison
        workload_df = generate_workload(num_packets=packets, duration=90, seed=seed)
        
        # Save workload
        workload_file = f"{output_dir}/workload_ep{ep}.csv"
        save_workload(workload_df, workload_file)
        
        # Extract features
        demand = generate_demand_vector(workload_df)
        features = extractor.extract_features(demand)
        features_tensor = torch.tensor(features, dtype=torch.float32, device=device)
        
        # --- TRAINED POLICY ---
        with torch.no_grad():
            trained_flows, _ = policy.get_action(features_tensor, k=8, deterministic=True)
        
        trained_result = runner.run_episode(
            workload_file=workload_file,
            critical_flows=trained_flows,
            output_dir=f"{output_dir}/trained_ep{ep}",
        )
        
        if trained_result['success']:
            trained_reward = compute_reward(trained_result['episode_summary'])
            trained_results.append({
                'reward': trained_reward,
                'queuing': trained_result['episode_summary'].get('mean_queuing_ms', 0),
                'drop_rate': trained_result['episode_summary'].get('drop_rate', 0),
                'flows': trained_flows,
            })
        
        # --- RANDOM POLICY ---
        rng = np.random.default_rng(seed)
        random_flows = rng.choice(NUM_FLOWS, size=8, replace=False).tolist()
        
        random_result = runner.run_episode(
            workload_file=workload_file,
            critical_flows=random_flows,
            output_dir=f"{output_dir}/random_ep{ep}",
        )
        
        if random_result['success']:
            random_reward = compute_reward(random_result['episode_summary'])
            random_results.append({
                'reward': random_reward,
                'queuing': random_result['episode_summary'].get('mean_queuing_ms', 0),
                'drop_rate': random_result['episode_summary'].get('drop_rate', 0),
                'flows': random_flows,
            })
        
        # Progress
        if trained_result['success'] and random_result['success']:
            print(f"[Ep {ep:2d}] Trained: R={trained_reward:+.3f}, Q={trained_results[-1]['queuing']:.0f}ms | "
                  f"Random: R={random_reward:+.3f}, Q={random_results[-1]['queuing']:.0f}ms")
        
        # Update history for next episode
        if trained_result['success']:
            extractor.update_history(
                trained_result['flow_summary'],
                trained_result['link_stats'],
                trained_result['episode_summary'],
            )
        
        # Cleanup
        runner.cleanup_cloudsim_results()
    
    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    if trained_results and random_results:
        trained_rewards = [r['reward'] for r in trained_results]
        random_rewards = [r['reward'] for r in random_results]
        trained_queuing = [r['queuing'] for r in trained_results]
        random_queuing = [r['queuing'] for r in random_results]
        trained_drops = [r['drop_rate'] for r in trained_results]
        random_drops = [r['drop_rate'] for r in random_results]
        
        print(f"\n{'Metric':<20} {'Trained':>12} {'Random':>12} {'Diff':>12}")
        print("-" * 60)
        print(f"{'Mean Reward':<20} {np.mean(trained_rewards):>12.4f} {np.mean(random_rewards):>12.4f} "
              f"{np.mean(trained_rewards) - np.mean(random_rewards):>+12.4f}")
        print(f"{'Mean Queuing (ms)':<20} {np.mean(trained_queuing):>12.1f} {np.mean(random_queuing):>12.1f} "
              f"{np.mean(trained_queuing) - np.mean(random_queuing):>+12.1f}")
        print(f"{'Mean Drop Rate':<20} {np.mean(trained_drops):>12.4f} {np.mean(random_drops):>12.4f} "
              f"{np.mean(trained_drops) - np.mean(random_drops):>+12.4f}")
        
        # Statistical significance
        from scipy import stats
        t_stat, p_value = stats.ttest_rel(trained_rewards, random_rewards)
        print(f"\nPaired t-test: t={t_stat:.3f}, p={p_value:.4f}")
        if p_value < 0.05:
            if np.mean(trained_rewards) > np.mean(random_rewards):
                print("→ Trained policy is SIGNIFICANTLY BETTER than random")
            else:
                print("→ Trained policy is SIGNIFICANTLY WORSE than random")
        else:
            print("→ No significant difference between trained and random")
        
        # Flow selection analysis
        print("\n" + "-" * 60)
        print("FLOW SELECTION ANALYSIS")
        print("-" * 60)
        
        trained_flow_counts = {}
        for r in trained_results:
            for f in r['flows']:
                trained_flow_counts[f] = trained_flow_counts.get(f, 0) + 1
        
        print("\nMost frequently selected flows by trained policy:")
        sorted_flows = sorted(trained_flow_counts.items(), key=lambda x: -x[1])[:10]
        for flow_id, count in sorted_flows:
            print(f"  Flow {flow_id:3d}: selected {count}/{len(trained_results)} times ({100*count/len(trained_results):.0f}%)")
    
    return trained_results, random_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained policy vs random")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--cloudsim-dir", type=str, default=".",
                       help="Path to CloudSimSDN directory")
    parser.add_argument("--episodes", type=int, default=20,
                       help="Number of evaluation episodes")
    parser.add_argument("--packets", type=int, default=300,
                       help="Packets per episode")
    parser.add_argument("--output-dir", type=str, default="eval_outputs",
                       help="Output directory")
    
    args = parser.parse_args()
    
    evaluate_policy(
        checkpoint_path=args.checkpoint,
        cloudsim_dir=args.cloudsim_dir,
        num_episodes=args.episodes,
        packets=args.packets,
        output_dir=args.output_dir,
    )