"""
cloudsim_trainer.py - Main training loop for CloudSim-in-the-loop RL.

Updated for Fat-Tree topology (16 hosts, 240 flows).

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

from workload_generator import (
    generate_workload, save_workload, generate_demand_vector, 
    get_workload_stats, load_workload, NUM_FLOWS, NUM_HOSTS
)
from episode_runner import EpisodeRunner, compute_reward
from feature_extractor import FeatureExtractor
from policy_network import PolicyNetwork, BatchReinforceTrainer


# =============================================================================
# Configuration
# =============================================================================

class TrainingConfig:
    """Training configuration with defaults for Fat-Tree topology."""
    
    # Episode settings
    num_episodes: int = 500
    packets_per_episode: int = 300        # High traffic for congestion
    episode_duration: float = 10.0        # Shorter duration = higher burst rate
    
    # Policy settings
    k_critical: int = 12                  # 12 of 60 = 20%
    num_features: int = 9
    num_flows: int = NUM_FLOWS            # 240 for fat-tree
    hidden_dims: list = [512, 256, 128]
    
    # Workload settings
    flow_selection: str = 'balanced'      # 'balanced', 'inter_pod', 'all' (legacy)
    traffic_model: str = 'hotspot'        # 'uniform', 'hotspot', 'gravity', 'skewed'
    
    # Training settings
    learning_rate: float = 3e-4           # Increased for batch updates
    baseline_decay: float = 0.99
    entropy_weight: float = 0.01
    max_grad_norm: float = 0.5
    
    # Batch training settings
    batch_size: int = 10                  # Episodes per batch update
    num_updates_per_batch: int = 3        # Gradient steps per batch
    normalize_advantages: bool = True
    
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
    save_episode_freq: int = 50          # Save episode data every N episodes (0 = never)
    
    # Paths
    cloudsim_dir: str = "."
    output_dir: str = "training_outputs"
    
    def __init__(self, **kwargs):
        # First, copy all class defaults to instance
        for attr in dir(self):
            if not attr.startswith('_') and not callable(getattr(self, attr)):
                setattr(self, attr, getattr(self, attr))
        # Then override with any provided kwargs
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
            num_flows=config.num_flows,
            num_features=config.num_features,
            hidden_dims=config.hidden_dims,
        ).to(self.device)
        
        self.trainer = BatchReinforceTrainer(
            self.policy,
            lr=config.learning_rate,
            batch_size=config.batch_size,
            num_updates_per_batch=config.num_updates_per_batch,
            entropy_weight=config.entropy_weight,
            max_grad_norm=config.max_grad_norm,
            normalize_advantages=config.normalize_advantages,
            total_episodes=config.num_episodes,
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
            json.dump(config.to_dict(), f, indent=2, default=str)
        
        print("CloudSimTrainer initialized")
        print(f"  Topology: Fat-Tree k=4 ({NUM_HOSTS} hosts, {NUM_FLOWS} flows)")
        print(f"  Device: {self.device}")
        print(f"  Output dir: {self.output_dir}")
        print(f"  Policy parameters: {sum(p.numel() for p in self.policy.parameters()):,}")
    
    def save_checkpoint(self, path: str, is_best: bool = False) -> None:
        """Save training checkpoint."""
        checkpoint = {
            'episode': self.episode,
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'baseline': self.trainer.baseline,
            'temperature': self.temperature,
            'best_reward': self.best_reward,
            'config': self.config.to_dict(),
        }
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.trainer.baseline = checkpoint['baseline']
        self.temperature = checkpoint['temperature']
        self.best_reward = checkpoint['best_reward']
        self.episode = checkpoint['episode']
        
        print(f"Loaded checkpoint from {path}")
        print(f"  Resuming from episode {self.episode}")
        print(f"  Best reward: {self.best_reward:.4f}")
    
    def run_episode(
        self,
        episode_id: int,
        existing_workload: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run a single training episode.
        
        Args:
            episode_id: Episode number
            existing_workload: Path to existing workload file (optional)
        
        Returns:
            Dictionary with episode results
        """
        if existing_workload:
            # Use existing workload file
            workload_path = Path(self.config.cloudsim_dir) / existing_workload
            episode_dir = f"episodes/ep_{episode_id:04d}"
            workload_df = load_workload(str(workload_path))
            workload_file = existing_workload
            should_save = True  # Always save when using existing workload
        else:
            # Determine if we should save this episode's data
            save_freq = self.config.save_episode_freq
            should_save = save_freq > 0 and (episode_id % save_freq == 0)
            
            # Use temp folder for non-saved episodes, permanent folder for saved ones
            if should_save:
                episode_dir = f"episodes/ep_{episode_id:04d}"
            else:
                episode_dir = "episodes/temp"
            
            # 1. Generate random workload
            workload_df = generate_workload(
                num_packets=self.config.packets_per_episode,
                duration=self.config.episode_duration,
                seed=None,  # Truly random
                traffic_model=self.config.traffic_model,
            )
            
            # Save workload
            workload_file = f"{episode_dir}/workload.csv"
            workload_path = self.output_dir / workload_file
            workload_path.parent.mkdir(parents=True, exist_ok=True)
            save_workload(workload_df, str(workload_path))
        
        # 2. Extract features
        demand_vector = generate_demand_vector(workload_df)
        features = self.feature_extractor.extract_features(demand_vector)
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        # 3. Get policy action (select K critical flows)
        # Note: Don't use no_grad here - we need gradients for REINFORCE
        logits = self.policy(features_tensor)
        
        # Sample with temperature
        selected_flows, log_prob = self.policy.sample_action(
            logits, 
            k=self.config.k_critical,
            temperature=self.temperature,
        )
        
        # 4. Run CloudSim episode
        output_dir = str(self.output_dir / episode_dir)
        results = self.episode_runner.run_episode(
            workload_file=str(workload_path),
            critical_flows=selected_flows,
            output_dir=output_dir,
            episode_id=episode_id,
        )
        
        # Clean up CloudSim's auto-generated result_* directories
        self.episode_runner.cleanup_cloudsim_results()
        
        if not results.get('success', False):
            print(f"  Episode {episode_id} FAILED: {results.get('error', 'Unknown error')}")
            # Clean up temp folder on failure too
            if not should_save:
                temp_dir = self.output_dir / "episodes" / "temp"
                if temp_dir.exists():
                    import shutil
                    shutil.rmtree(temp_dir)
            return {'success': False, 'reward': -10.0}
        
        # 5. Compute reward
        episode_summary = results.get('episode_summary', {})
        reward = compute_reward(
            episode_summary=episode_summary,
            queuing_weight=self.config.queuing_weight,
            drop_penalty=self.config.drop_penalty,
        )
        
        # 6. Update feature extractor with flow statistics
        flow_summary = results.get('flow_summary')
        if flow_summary is not None and not flow_summary.empty:
            self.feature_extractor.update_historical_features(flow_summary)
        
        # 7. Clean up temp episode folder if not saving
        if not should_save:
            temp_dir = self.output_dir / "episodes" / "temp"
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir)
        
        return {
            'success': True,
            'reward': reward,
            'log_prob': log_prob,
            'selected_flows': selected_flows,
            'features_tensor': features_tensor,
            'episode_summary': episode_summary,
            'demand_vector': demand_vector,
            'wall_time_ms': results.get('wall_time_ms', 0),
        }
    
    def train(self, start_episode: int = 0, end_episode: int = None) -> None:
        """
        Run training loop with batch REINFORCE updates.
        
        Args:
            start_episode: Starting episode number
            end_episode: Ending episode number (exclusive)
        """
        if end_episode is None:
            end_episode = self.config.num_episodes
        
        print(f"Starting training from episode {start_episode} to {end_episode}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Updates per batch: {self.config.num_updates_per_batch}")
        print("=" * 70)
        
        for episode_id in range(start_episode, end_episode):
            self.episode = episode_id
            
            # Run episode
            results = self.run_episode(episode_id)
            
            if not results.get('success', False):
                continue
            
            reward = results['reward']
            log_prob = results['log_prob']
            features_tensor = results['features_tensor']
            selected_flows = results['selected_flows']
            episode_summary = results.get('episode_summary', {})
            
            # Store episode for batch update
            self.trainer.store_episode(
                features=features_tensor,
                selected_flows=selected_flows,
                log_prob=log_prob,
                reward=reward,
                temperature=self.temperature,
            )
            
            # Batch update when buffer is full
            loss = 0.0
            if self.trainer.should_update():
                update_metrics = self.trainer.update()
                loss = update_metrics.get('loss', 0.0)
                print(f"    [Batch Update #{update_metrics.get('num_updates', 0)}] "
                      f"loss={loss:.4f} batch_R={update_metrics.get('batch_reward_mean', 0):.4f}")
            
            # Update temperature
            self.temperature = max(
                self.config.temperature_end,
                self.temperature * self.config.temperature_decay
            )
            
            # Log metrics
            metrics = {
                'reward': reward,
                'mean_queuing_ms': episode_summary.get('mean_queuing_ms', 0),
                'drop_rate': episode_summary.get('drop_rate', 0),
                'loss': loss,
                'baseline': self.trainer.get_baseline(),
                'temperature': self.temperature,
                'wall_time_ms': results.get('wall_time_ms', 0),
                'total_packets': episode_summary.get('total_packets', 0),
                'num_flows_active': episode_summary.get('num_flows_active', 0),
            }
            self.logger.log(episode_id, metrics)
            
            # Console output
            stats = self.logger.get_recent_stats(10)
            print(f"[Ep {episode_id:4d}] R={reward:.4f} Q={metrics['mean_queuing_ms']:.1f}ms "
                  f"D={metrics['drop_rate']:.4f} T={self.temperature:.3f} | "
                  f"Avg10: R={stats.get('mean_reward', 0):.4f}")
            
            # Track best model
            if reward > self.best_reward:
                self.best_reward = reward
                self.save_checkpoint(
                    str(self.checkpoint_dir / "best_model.pt"),
                    is_best=True
                )
                print(f"    ★ New best reward: {reward:.6f}")
            
            # Periodic checkpoint
            if (episode_id + 1) % self.config.checkpoint_freq == 0:
                self.save_checkpoint(
                    str(self.checkpoint_dir / f"checkpoint_ep{episode_id:04d}.pt")
                )
        
        # Final checkpoint
        self.save_checkpoint(str(self.checkpoint_dir / "final_model.pt"))
        
        print("=" * 70)
        print("Training complete!")
        print(f"  Best reward: {self.best_reward:.6f}")
        print(f"  Final checkpoint: {self.checkpoint_dir}/final_model.pt")


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_policy(
    trainer: CloudSimTrainer,
    num_episodes: int = 50,
    use_random_baseline: bool = False,
) -> Dict[str, float]:
    """
    Evaluate a trained policy.
    
    Args:
        trainer: CloudSimTrainer with loaded policy
        num_episodes: Number of evaluation episodes
        use_random_baseline: If True, use random policy instead
    
    Returns:
        Dictionary with evaluation statistics
    """
    rewards = []
    queuing_delays = []
    drop_rates = []
    
    # Set to eval mode (no gradient)
    trainer.policy.eval()
    
    for ep in range(num_episodes):
        results = trainer.run_episode(
            episode_id=10000 + ep,  # Offset to avoid training episodes
        )
        
        if results.get('success', False):
            rewards.append(results['reward'])
            summary = results.get('episode_summary', {})
            queuing_delays.append(summary.get('mean_queuing_ms', 0))
            drop_rates.append(summary.get('drop_rate', 0))
    
    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_queuing_ms': np.mean(queuing_delays),
        'mean_drop_rate': np.mean(drop_rates),
        'num_episodes': len(rewards),
    }


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="CFR-RL CloudSim Training (Fat-Tree)")
    
    # Required
    parser.add_argument("--cloudsim-dir", type=str, required=True,
                       help="Path to CloudSimSDN project directory")
    
    # Training settings
    parser.add_argument("--episodes", type=int, default=500,
                       help="Number of training episodes")
    parser.add_argument("--packets", type=int, default=300,
                       help="Packets per episode (default: 300)")
    parser.add_argument("--duration", type=float, default=10.0,
                       help="Episode duration in seconds (default: 10)")
    parser.add_argument("--k-critical", type=int, default=12,
                       help="Number of critical flows to select (K)")
    parser.add_argument("--lr", type=float, default=3e-4,
                       help="Learning rate (default: 3e-4 for batch training)")
    parser.add_argument("--batch-size", type=int, default=10,
                       help="Episodes per batch update (default: 10)")
    parser.add_argument("--flow-selection", type=str, default='balanced',
                       choices=['balanced', 'inter_pod', 'all'],
                       help="Flow selection strategy (legacy, use --traffic-model instead)")
    parser.add_argument("--traffic-model", type=str, default='hotspot',
                       choices=['uniform', 'hotspot', 'gravity', 'skewed'],
                       help="Traffic model: uniform, hotspot, gravity, skewed (default: hotspot)")
    parser.add_argument("--save-episode-freq", type=int, default=50,
                       help="Save episode data every N episodes (0=never, default=50)")
    
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
        k_critical=args.k_critical,
        traffic_model=args.traffic_model,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        save_episode_freq=args.save_episode_freq,
        cloudsim_dir=args.cloudsim_dir,
        output_dir=args.output_dir,
    )
    
    # Create trainer
    trainer = CloudSimTrainer(config)
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Evaluation mode
    if args.eval_only:
        if not args.resume:
            print("Error: --eval-only requires --resume")
            return
        
        print("Running evaluation...")
        stats = evaluate_policy(trainer, num_episodes=50)
        print("\nEvaluation Results:")
        print(f"  Mean reward: {stats['mean_reward']:.4f} ± {stats['std_reward']:.4f}")
        print(f"  Mean queuing: {stats['mean_queuing_ms']:.2f} ms")
        print(f"  Mean drop rate: {stats['mean_drop_rate']:.4f}")
        return
    
    # Training mode
    start_ep = trainer.episode if args.resume else 0
    trainer.train(start_episode=start_ep, end_episode=args.episodes)


if __name__ == "__main__":
    main()