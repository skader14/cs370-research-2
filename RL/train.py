"""
Main training script for CFR-RL.
"""
import argparse
import json
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from topology import Topology
from trainer import REINFORCETrainer


def plot_training_curves(stats: list, save_path: str):
    """Plot training curves."""
    rewards = [s["reward"] for s in stats]
    losses = [s["loss"] for s in stats]
    
    # Smooth with moving average
    window = 50
    rewards_smooth = np.convolve(rewards, np.ones(window)/window, mode='valid')
    losses_smooth = np.convolve(losses, np.ones(window)/window, mode='valid')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(rewards, alpha=0.3, label='Raw')
    ax1.plot(range(window-1, len(rewards)), rewards_smooth, label='Smoothed')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Reward (1/max_util)')
    ax1.set_title('Training Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(losses, alpha=0.3, label='Raw')
    ax2.plot(range(window-1, len(losses)), losses_smooth, label='Smoothed')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Training curves saved to {save_path}")


def plot_comparison(results: dict, save_path: str):
    """Plot comparison bar chart."""
    methods = list(results.keys())
    means = [results[m]["mean"] for m in methods]
    stds = [results[m]["std"] for m in methods]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = range(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, 
                  color=['green', 'blue', 'orange', 'red'])
    
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel('Max Link Utilization')
    ax.set_title('Comparison of Routing Methods')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{mean:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Comparison plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train CFR-RL policy")
    parser.add_argument("--n-edge", type=int, default=4, 
                       help="Number of edge switches")
    parser.add_argument("--hosts-per-edge", type=int, default=4,
                       help="Hosts per edge switch")
    parser.add_argument("--k-critical", type=int, default=5,
                       help="Number of critical flows to select")
    parser.add_argument("--iterations", type=int, default=1000,
                       help="Training iterations")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=128,
                       help="Hidden dimension")
    parser.add_argument("--entropy-coef", type=float, default=0.01,
                       help="Entropy coefficient")
    parser.add_argument("--output-dir", type=str, default="./results",
                       help="Output directory")
    parser.add_argument("--eval-episodes", type=int, default=50,
                       help="Evaluation episodes")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("CFR-RL Training")
    print("=" * 60)
    print(f"Topology: {args.n_edge} edge switches, {args.hosts_per_edge} hosts/edge")
    print(f"Total hosts: {args.n_edge * args.hosts_per_edge}")
    print(f"Total flows: {args.n_edge * args.hosts_per_edge * (args.n_edge * args.hosts_per_edge - 1)}")
    print(f"K critical: {args.k_critical}")
    print(f"Iterations: {args.iterations}")
    print("=" * 60)
    
    # Initialize
    topo = Topology(n_edge=args.n_edge, hosts_per_edge=args.hosts_per_edge)
    trainer = REINFORCETrainer(
        topo, 
        k_critical=args.k_critical,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        entropy_coef=args.entropy_coef,
    )
    
    # Train
    print("\nStarting training...")
    start_time = time.time()
    
    stats = trainer.train(
        n_iterations=args.iterations,
        traffic_method="bimodal",
        log_interval=100,
        save_path=os.path.join(args.output_dir, "policy.pt"),
        n_elephant=8,           # More elephants
        elephant_demand=700e6,  # 700 Mbps each  
        mice_demand=60e6,       # Bigger mice
        sparsity=0.5,           # More flows
    )
    
    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time:.1f} seconds")
    
    # Plot training curves
    plot_training_curves(stats, os.path.join(args.output_dir, "training_curves.png"))
    
    # Evaluate
    print("\nEvaluating policy...")
    results = trainer.evaluate(
        n_episodes=args.eval_episodes,
        traffic_method="bimodal",
        compare_baselines=True,
        n_elephant=8,
        elephant_demand=700e6,
        mice_demand=60e6,
        sparsity=0.5,
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"{'Method':<20} {'Mean Util':>12} {'Std':>10} {'vs Single':>12} {'vs ECMP':>10}")
    print("-" * 70)

    single_mean = results["single_path"]["mean"]
    ecmp_mean = results["ecmp"]["mean"]

    # Order: worst to best
    method_order = ["single_path", "random_k", "top_k", "cfr_rl", "ecmp"]

    for method in method_order:
        stats = results[method]
        
        # Improvement vs single-path baseline
        imp_vs_single = (single_mean - stats["mean"]) / single_mean * 100
        
        # Gap to ECMP (lower is better - how close to optimal)
        if method == "ecmp":
            gap_to_ecmp = "optimal"
        elif method == "single_path":
            gap_to_ecmp = "baseline"
        else:
            # How much of the possible improvement did we capture?
            possible_imp = single_mean - ecmp_mean
            actual_imp = single_mean - stats["mean"]
            if possible_imp > 0:
                gap_to_ecmp = f"{actual_imp/possible_imp*100:.1f}%"
            else:
                gap_to_ecmp = "N/A"
        
        print(f"{method:<20} {stats['mean']:>12.4f} {stats['std']:>10.4f} "
            f"{imp_vs_single:>+11.1f}% {gap_to_ecmp:>10}")

    print("-" * 70)
    print("Note: 'vs ECMP' shows what % of possible improvement was captured")
    print("      (ECMP is optimal but requires rules for ALL flows)")
    
    # Plot comparison
    plot_comparison(results, os.path.join(args.output_dir, "comparison.png"))
    
    # Save results to JSON
    results_file = os.path.join(args.output_dir, "results.json")
    with open(results_file, "w") as f:
        json.dump({
            "config": vars(args),
            "training_time_seconds": train_time,
            "evaluation_results": results,
        }, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model saved to: {os.path.join(args.output_dir, 'policy.pt')}")
    print("=" * 60)


if __name__ == "__main__":
    main()