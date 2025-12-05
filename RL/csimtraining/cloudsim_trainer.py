"""
cloudsim_trainer.py - Main training loop for CloudSim-in-the-loop RL.

This script:
1. Generates random workloads for each episode
2. Uses the policy network to select K critical flows
3. Runs CloudSim as a subprocess
4. Reads results and computes reward
5. Updates policy using REINFORCE

Usage:
    python cloudsim_trainer.py --episodes 500 --cloudsim-dir /path/to/cloudsimsdn

    # Resume from checkpoint
    python cloudsim_trainer.py --resume checkpoints/latest.pt
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import torch
import pandas as pd

from workload_generator import generate_workload, save_workload, generate_demand_vector, get_workload_stats
from episode_runner import EpisodeRunner, compute_reward
from feature_extractor import FeatureExtractor, NUM_FLOWS
from policy_network import PolicyNetwork, FlowWisePolicyNetwork, ReinforceTrainer, K_CRITICAL


# =============================================================================
# Configuration
# =============================================================================

class TrainingConfig:
    """Training configuration with defaults."""
    
    # Episode settings
    num_episodes: int = 500
    packets_per_episode: int = 300
    episode_duration: float = 90.0
    
    # Policy settings
    k_critical: int = 8
    num_features: int = 9
    hidden_dims: list = [512, 256, 128]
    
    # Training settings
    learning_rate: float = 1e-4
    baseline_decay: float = 0.99
    entropy_weight: float = 0.01
    max_grad_norm: float = 1.0
    
    # Temperature schedule
    temperature_start: float = 1.0
    temperature_end: float = 0.1
    temperature_decay: float = 0.995
    
    # Reward settings
    queuing_weight: float = 1.0
    drop_penalty: float = 10.0
    
    # Checkpointing
    checkpoint_freq: int = 50
    eval_freq: int = 10
    
    # Paths
    cloudsim_dir: str = "."
    output_dir: str = "training_outputs"
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def to_dict(self) -> dict:
        return {k: v for k, v in vars(self).items() if not k.startswith('_')}


# =============================================================================
# Training Metrics Logger
# =============================================================================

class MetricsLogger:
    """Log training metrics to CSV and console."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.csv_path = self.output_dir / "training_metrics.csv"
        self.metrics_buffer = []
        
        # Write header
        with open(self.csv_path, 'w') as f:
            f.write("episode,timestamp,reward,mean_queuing_ms,drop_rate,loss,baseline,"
                   "temperature,wall_time_ms,packets,active_flows\n")
    
    def log(self, episode: int, metrics: dict) -> None:
        """Log metrics for one episode."""
        timestamp = datetime.now().isoformat()
        
        row = (
            f"{episode},{timestamp},{metrics.get('reward', 0):.6f},"
            f"{metrics.get('mean_queuing_ms', 0):.4f},{metrics.get('drop_rate', 0):.6f},"
            f"{metrics.get('loss', 0):.6f},{metrics.get('baseline', 0):.6f},"
            f"{metrics.get('temperature', 1.0):.4f},{metrics.get('wall_time_ms', 0)},"
            f"{metrics.get('total_packets', 0)},{metrics.get('num_flows_active', 0)}\n"
        )
        
        with open(self.csv_path, 'a') as f:
            f.write(row)
        
        self.metrics_buffer.append(metrics)
    
    def get_recent_stats(self, n: int = 10) -> dict:
        """Get statistics over last n episodes."""
        recent = self.metrics_buffer[-n:] if len(self.metrics_buffer) >= n else self.metrics_buffer
        if not recent:
            return {}
        
        rewards = [m.get('reward', 0) for m in recent]
        queuing = [m.get('mean_queuing_ms', 0) for m in recent]
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_queuing': np.mean(queuing),
            'best_reward': max(rewards),
        }


# =============================================================================
# Main Trainer
# =============================================================================

class CloudSimTrainer:
    """
    Main trainer for CloudSim-in-the-loop RL.
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create output directories
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.episodes_dir = self.output_dir / "episodes"
        self.episodes_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.policy = PolicyNetwork(
            num_flows=NUM_FLOWS,
            num_features=config.num_features,
            hidden_dims=config.hidden_dims,
        ).to(self.device)
        
        self.trainer = ReinforceTrainer(
            self.policy,
            lr=config.learning_rate,
            baseline_decay=config.baseline_decay,
            entropy_weight=config.entropy_weight,
            max_grad_norm=config.max_grad_norm,
        )
        
        self.feature_extractor = FeatureExtractor(random_cold_start=True)
        
        self.episode_runner = EpisodeRunner(
            cloudsim_dir=config.cloudsim_dir,
            timeout=120,
            verbose=True,
        )
        
        self.logger = MetricsLogger(str(self.output_dir))
        
        # Temperature schedule
        self.temperature = config.temperature_start
        
        # Best model tracking
        self.best_reward = float('-inf')
        
        # Episode counter
        self.episode = 0
        
        # Save config
        config_path = self.output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        
        print(f"CloudSimTrainer initialized")
        print(f"  Device: {self.device}")
        print(f"  Output dir: {self.output_dir}")
        print(f"  Policy parameters: {sum(p.numel() for p in self.policy.parameters()):,}")
    
    def train(self, num_episodes: Optional[int] = None) -> None:
        """Run training loop."""
        num_episodes = num_episodes or self.config.num_episodes
        start_episode = self.episode
        
        print(f"\nStarting training from episode {start_episode} to {start_episode + num_episodes}")
        print("=" * 70)
        
        for ep in range(start_episode, start_episode + num_episodes):
            self.episode = ep
            metrics = self._run_episode(ep)
            
            if metrics['success']:
                # Log metrics
                self.logger.log(ep, metrics)
                
                # Print progress
                self._print_progress(ep, metrics)
                
                # Checkpoint
                if (ep + 1) % self.config.checkpoint_freq == 0:
                    self._save_checkpoint(f"checkpoint_ep{ep+1}.pt")
                
                # Track best
                if metrics['reward'] > self.best_reward:
                    self.best_reward = metrics['reward']
                    self._save_checkpoint("best_model.pt")
                    print(f"    ★ New best reward: {self.best_reward:.6f}")
            else:
                print(f"  Episode {ep} FAILED: {metrics.get('error', 'Unknown error')}")
            
            # Decay temperature
            self.temperature = max(
                self.config.temperature_end,
                self.temperature * self.config.temperature_decay
            )
        
        # Final checkpoint
        self._save_checkpoint("final_model.pt")
        print("\n" + "=" * 70)
        print("Training complete!")
        print(f"  Best reward: {self.best_reward:.6f}")
        print(f"  Final checkpoint: {self.checkpoint_dir}/final_model.pt")
    
    def _run_episode(self, episode_id: int) -> Dict[str, Any]:
        """Run a single training episode."""
        
        # 1. Generate random workload
        workload_df = generate_workload(
            num_packets=self.config.packets_per_episode,
            duration=self.config.episode_duration,
            seed=None,  # Truly random
        )
        
        # Save workload
        episode_dir = f"episodes/ep_{episode_id:04d}"
        workload_file = f"{episode_dir}/workload.csv"
        workload_path = self.output_dir / workload_file
        workload_path.parent.mkdir(parents=True, exist_ok=True)
        save_workload(workload_df, str(workload_path))
        
        # 2. Extract features
        demand = generate_demand_vector(workload_df)
        features = self.feature_extractor.extract_features(demand)
        features_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
        
        # 3. Select critical flows
        selected_flows, log_prob = self.policy.get_action(
            features_tensor,
            k=self.config.k_critical,
            temperature=self.temperature,
        )
        
        # 4. Run CloudSim episode
        results = self.episode_runner.run_episode(
            workload_file=str(workload_path.relative_to(Path(self.config.cloudsim_dir))),
            critical_flows=selected_flows,
            output_dir=str((self.output_dir / episode_dir).relative_to(Path(self.config.cloudsim_dir))),
            episode_id=episode_id,
        )
        
        if not results['success']:
            return results
        
        # 5. Compute reward
        episode_summary = results['episode_summary']
        reward = compute_reward(
            episode_summary,
            queuing_weight=self.config.queuing_weight,
            drop_penalty=self.config.drop_penalty,
        )
        
        # 6. Update policy
        update_metrics = self.trainer.update(
            features_tensor,
            selected_flows,
            reward,
            temperature=self.temperature,
        )
        
        # 7. Update feature extractor history
        self.feature_extractor.update_history(
            results['flow_summary'],
            results['link_stats'],
            episode_summary,
        )
        
        # 8. Compile metrics
        metrics = {
            'success': True,
            'reward': reward,
            'mean_queuing_ms': episode_summary.get('mean_queuing_ms', 0),
            'max_queuing_ms': episode_summary.get('max_queuing_ms', 0),
            'drop_rate': episode_summary.get('drop_rate', 0),
            'total_packets': episode_summary.get('total_packets', 0),
            'num_flows_active': episode_summary.get('num_flows_active', 0),
            'temperature': self.temperature,
            'wall_time_ms': results['wall_time_ms'],
            'selected_flows': selected_flows,
            **update_metrics,
        }
        
        # Clean up episode files to save space (keep every 100th for debugging)
        if episode_id % 100 != 0:
            self.episode_runner.cleanup_episode(
                str((self.output_dir / episode_dir).relative_to(Path(self.config.cloudsim_dir)))
            )
        
        # Always clean up CloudSim's auto-generated result directories
        self.episode_runner.cleanup_cloudsim_results()
        
        return metrics
    
    def _print_progress(self, episode: int, metrics: dict) -> None:
        """Print training progress."""
        recent = self.logger.get_recent_stats(10)
        
        print(f"[Ep {episode:4d}] "
              f"R={metrics['reward']:+.4f} "
              f"Q={metrics['mean_queuing_ms']:.1f}ms "
              f"D={metrics['drop_rate']:.4f} "
              f"T={metrics['temperature']:.3f} "
              f"| Avg10: R={recent.get('mean_reward', 0):+.4f}")
    
    def _save_checkpoint(self, filename: str) -> None:
        """Save training checkpoint."""
        path = self.checkpoint_dir / filename
        
        checkpoint = {
            'episode': self.episode,
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'baseline': self.trainer.baseline,
            'temperature': self.temperature,
            'best_reward': self.best_reward,
            'config': self.config.to_dict(),
        }
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.trainer.baseline = checkpoint['baseline']
        self.trainer.baseline_initialized = True
        self.temperature = checkpoint['temperature']
        self.best_reward = checkpoint['best_reward']
        self.episode = checkpoint['episode']
        
        print(f"Loaded checkpoint from {filepath}")
        print(f"  Episode: {self.episode}")
        print(f"  Best reward: {self.best_reward:.6f}")
        print(f"  Temperature: {self.temperature:.4f}")
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate current policy without training."""
        print(f"\nEvaluating over {num_episodes} episodes...")
        
        rewards = []
        queuing_times = []
        
        self.policy.eval()
        
        with torch.no_grad():
            for ep in range(num_episodes):
                # Generate workload
                workload_df = generate_workload(
                    num_packets=self.config.packets_per_episode,
                    duration=self.config.episode_duration,
                )
                
                # Extract features
                demand = generate_demand_vector(workload_df)
                features = self.feature_extractor.extract_features(demand)
                features_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
                
                # Deterministic action
                selected_flows, _ = self.policy.get_action(
                    features_tensor,
                    k=self.config.k_critical,
                    deterministic=True,
                )
                
                # Save workload and run
                workload_file = f"eval_ep_{ep}.csv"
                save_workload(workload_df, str(self.output_dir / workload_file))
                
                results = self.episode_runner.run_episode(
                    workload_file=str((self.output_dir / workload_file).relative_to(Path(self.config.cloudsim_dir))),
                    critical_flows=selected_flows,
                    output_dir=str((self.output_dir / f"eval_ep_{ep}").relative_to(Path(self.config.cloudsim_dir))),
                )
                
                if results['success']:
                    reward = compute_reward(results['episode_summary'])
                    rewards.append(reward)
                    queuing_times.append(results['episode_summary'].get('mean_queuing_ms', 0))
        
        self.policy.train()
        
        stats = {
            'mean_reward': np.mean(rewards) if rewards else 0,
            'std_reward': np.std(rewards) if rewards else 0,
            'mean_queuing_ms': np.mean(queuing_times) if queuing_times else 0,
            'success_rate': len(rewards) / num_episodes,
        }
        
        print(f"Evaluation results:")
        print(f"  Mean reward: {stats['mean_reward']:.6f} ± {stats['std_reward']:.6f}")
        print(f"  Mean queuing: {stats['mean_queuing_ms']:.2f} ms")
        print(f"  Success rate: {stats['success_rate']:.1%}")
        
        return stats


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="CloudSim-in-the-loop RL Training")
    
    # Required arguments
    parser.add_argument("--cloudsim-dir", type=str, default=".",
                       help="Path to CloudSimSDN project directory")
    
    # Training settings
    parser.add_argument("--episodes", type=int, default=500,
                       help="Number of training episodes")
    parser.add_argument("--packets", type=int, default=300,
                       help="Packets per episode")
    parser.add_argument("--duration", type=float, default=90.0,
                       help="Episode duration (simulated seconds)")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="training_outputs",
                       help="Output directory for checkpoints and logs")
    
    # Resume
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint file")
    
    # Evaluation only
    parser.add_argument("--eval-only", action="store_true",
                       help="Only run evaluation (requires --resume)")
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        num_episodes=args.episodes,
        packets_per_episode=args.packets,
        episode_duration=args.duration,
        learning_rate=args.lr,
        cloudsim_dir=args.cloudsim_dir,
        output_dir=args.output_dir,
    )
    
    # Create trainer
    trainer = CloudSimTrainer(config)
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train or evaluate
    if args.eval_only:
        if not args.resume:
            print("ERROR: --eval-only requires --resume to specify model checkpoint")
            return
        trainer.evaluate(num_episodes=20)
    else:
        trainer.train()


if __name__ == "__main__":
    main()