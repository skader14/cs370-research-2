"""
cloudsim_trainer.py - Main training loop for CloudSim-in-the-loop RL.

Updated for Fat-Tree topology (16 hosts, 240 flows).
Now supports both REINFORCE and PPO algorithms.
Supports dense rewards via time-windowed packet analysis.

This script:
1. Generates random workloads for each episode
2. Uses the policy network to select K critical flows
3. Runs CloudSim as a subprocess
4. Reads results and computes reward (sparse or dense)
5. Updates policy using PPO (default) or REINFORCE

Usage:
    # PPO training with sparse rewards
    python cloudsim_trainer.py --episodes 500 --cloudsim-dir /path/to/cloudsimsdn
    
    # PPO training with dense rewards (recommended)
    python cloudsim_trainer.py --episodes 500 --dense-rewards --cloudsim-dir /path/to/cloudsimsdn
    
    # REINFORCE training (for comparison)
    python cloudsim_trainer.py --episodes 500 --algorithm reinforce --cloudsim-dir /path/to/cloudsimsdn

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
from episode_runner import (
    EpisodeRunner, compute_reward, compute_dense_rewards,
    compute_dense_reward_from_results
)
from feature_extractor import FeatureExtractor
from policy_network import (
    PolicyNetwork, ValueNetwork, PPOTrainer, BatchReinforceTrainer,
    create_ppo_networks
)


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
    
    # Algorithm selection
    algorithm: str = 'ppo'                # 'ppo' or 'reinforce'
    
    # Baseline evaluation mode
    baseline: str = 'none'                # 'none', 'random', 'topk-demand', 'topk-queuing', 'ecmp'
    
    # Common training settings
    learning_rate: float = 3e-4           # Policy learning rate
    baseline_decay: float = 0.99
    entropy_weight: float = 0.01
    max_grad_norm: float = 0.5
    
    # Batch training settings
    batch_size: int = 10                  # Episodes per batch update
    normalize_advantages: bool = True
    
    # PPO-specific settings
    lr_value: float = 1e-3                # Value network learning rate (higher than policy)
    ppo_epochs: int = 4                   # Gradient updates per batch
    clip_epsilon: float = 0.2             # PPO clipping parameter
    value_loss_coef: float = 0.5          # Weight for value loss
    gae_lambda: float = 0.95              # GAE lambda
    target_kl: float = 0.02               # KL early stopping threshold (None to disable)
    
    # REINFORCE-specific settings (for comparison)
    num_updates_per_batch: int = 3        # Gradient steps per batch
    
    # Temperature schedule
    temperature_start: float = 1.0
    temperature_end: float = 0.1
    temperature_decay: float = 0.995
    min_temperature: float = 0.1          # Floor for temperature (can be > temperature_end)
    
    # Reward settings
    queuing_weight: float = 1.0
    drop_penalty: float = 10.0
    
    # Dense reward settings
    use_dense_rewards: bool = False       # Use windowed dense rewards
    dense_window_size: float = 1.0        # Window size in seconds
    dense_gamma: float = 0.99             # Discount factor for dense rewards
    dense_critical_only: bool = True      # Only use critical flow packets
    
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
    
    def __init__(self, output_dir: str, algorithm: str = 'ppo', dense_rewards: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.algorithm = algorithm
        self.dense_rewards = dense_rewards
        
        self.csv_path = self.output_dir / "training_metrics.csv"
        self.metrics_buffer = []
        
        # Write header (includes dense reward columns)
        header = (
            "episode,timestamp,reward,mean_queuing_ms,drop_rate,loss,"
            "policy_loss,value_loss,entropy,kl_div,baseline,"
            "temperature,wall_time_ms,packets,active_flows,"
            "num_windows,critical_packets,dense_reward_undiscounted\n"
        )
        with open(self.csv_path, 'w') as f:
            f.write(header)
    
    def log(self, episode: int, metrics: dict) -> None:
        """Log metrics for one episode."""
        timestamp = datetime.now().isoformat()
        
        row = (
            f"{episode},{timestamp},{metrics.get('reward', 0):.6f},"
            f"{metrics.get('mean_queuing_ms', 0):.4f},{metrics.get('drop_rate', 0):.6f},"
            f"{metrics.get('loss', 0):.6f},"
            f"{metrics.get('policy_loss', 0):.6f},{metrics.get('value_loss', 0):.6f},"
            f"{metrics.get('entropy', 0):.6f},{metrics.get('kl_divergence', 0):.6f},"
            f"{metrics.get('baseline', 0):.6f},"
            f"{metrics.get('temperature', 1.0):.4f},{metrics.get('wall_time_ms', 0)},"
            f"{metrics.get('total_packets', 0)},{metrics.get('num_flows_active', 0)},"
            f"{metrics.get('num_windows', 0)},{metrics.get('critical_packets', 0)},"
            f"{metrics.get('dense_reward_undiscounted', 0):.6f}\n"
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
# Baseline Selection Functions
# =============================================================================

def select_baseline_flows(baseline_type: str, k: int, demand_vector: np.ndarray = None,
                          prev_queuing: np.ndarray = None, num_flows: int = NUM_FLOWS) -> list:
    """
    Select K critical flows using a baseline heuristic.
    
    Args:
        baseline_type: One of 'random', 'topk-demand', 'topk-queuing', 'ecmp', 'all-critical'
        k: Number of flows to select (ignored for 'ecmp' and 'all-critical')
        demand_vector: Traffic demand for each flow (required for topk-demand)
        prev_queuing: Previous episode's queuing delay per flow (required for topk-queuing)
        num_flows: Total number of flows
        
    Returns:
        List of flow indices
    """
    if baseline_type == 'ecmp':
        # No critical flows - everything uses ECMP
        return []
    
    elif baseline_type == 'all-critical':
        # ALL flows are critical - theoretical upper bound
        # This is what full TE optimization would do
        return list(range(num_flows))
    
    elif baseline_type == 'random':
        # Uniform random selection
        return np.random.choice(num_flows, k, replace=False).tolist()
    
    elif baseline_type == 'topk-demand':
        # Select K highest-demand flows
        if demand_vector is None:
            raise ValueError("topk-demand baseline requires demand_vector")
        return np.argsort(demand_vector)[-k:].tolist()
    
    elif baseline_type == 'topk-queuing':
        # Select K flows with highest previous queuing delay
        if prev_queuing is None:
            # Fall back to random if no history
            return np.random.choice(num_flows, k, replace=False).tolist()
        return np.argsort(prev_queuing)[-k:].tolist()
    
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")


def compute_flow_overlap(flows_a: list, flows_b: list) -> dict:
    """
    Compute overlap between two flow selections.
    
    Returns:
        Dictionary with overlap metrics
    """
    set_a = set(flows_a)
    set_b = set(flows_b)
    
    intersection = set_a & set_b
    union = set_a | set_b
    
    return {
        'overlap_count': len(intersection),
        'overlap_pct': len(intersection) / max(len(set_a), 1) * 100,
        'jaccard': len(intersection) / max(len(union), 1),
        'only_in_a': len(set_a - set_b),
        'only_in_b': len(set_b - set_a),
    }


def compute_network_disturbance(selected_flows: list, prev_selected_flows: list,
                                 demand_vector: np.ndarray = None) -> dict:
    """
    Compute network disturbance metrics.
    
    Zhang et al.'s key insight: minimize disturbance while achieving near-optimal performance.
    
    Args:
        selected_flows: Currently selected critical flows
        prev_selected_flows: Previously selected critical flows
        demand_vector: Traffic demand per flow (for traffic-weighted metrics)
        
    Returns:
        Dictionary with disturbance metrics
    """
    curr_set = set(selected_flows)
    prev_set = set(prev_selected_flows) if prev_selected_flows else set()
    
    # Flows that changed status
    newly_critical = curr_set - prev_set  # Were ECMP, now critical
    no_longer_critical = prev_set - curr_set  # Were critical, now ECMP
    
    # Basic disturbance metrics
    disturbance = {
        'num_critical': len(selected_flows),
        'num_changes': len(newly_critical) + len(no_longer_critical),
        'newly_critical': len(newly_critical),
        'no_longer_critical': len(no_longer_critical),
        'stability_pct': len(curr_set & prev_set) / max(len(curr_set), 1) * 100 if curr_set else 100,
    }
    
    # Traffic-weighted disturbance (if demand available)
    if demand_vector is not None and len(demand_vector) > 0:
        total_demand = np.sum(demand_vector)
        critical_demand = sum(demand_vector[f] for f in selected_flows if f < len(demand_vector))
        changed_demand = sum(demand_vector[f] for f in (newly_critical | no_longer_critical) 
                            if f < len(demand_vector))
        
        disturbance['critical_traffic_pct'] = critical_demand / max(total_demand, 1) * 100
        disturbance['changed_traffic_pct'] = changed_demand / max(total_demand, 1) * 100
    
    return disturbance


def select_trained_policy_flows(trainer, features_tensor: torch.Tensor, 
                                 temperature: float = 0.15) -> list:
    """
    Select K critical flows using the trained policy network.
    
    Args:
        trainer: CloudSimTrainer with loaded policy
        features_tensor: Feature tensor for current state
        temperature: Sampling temperature (use low for exploitation)
        
    Returns:
        List of K flow indices
    """
    with torch.no_grad():
        logits = trainer.policy(features_tensor)
        selected_flows, _ = trainer.policy.sample_action(
            logits,
            k=trainer.config.k_critical,
            temperature=temperature,
        )
    return selected_flows


def run_single_episode_all_methods(trainer, workload_df, workload_path, episode_id: int,
                                   methods: list, prev_queuing: np.ndarray = None,
                                   prev_selected: dict = None,
                                   temperature: float = 0.15,
                                   save_episode: bool = False) -> dict:
    """
    Run a single episode with all methods using the SAME workload.
    
    This ensures fair comparison by eliminating workload variance.
    
    Args:
        trainer: CloudSimTrainer instance
        workload_df: Workload DataFrame
        workload_path: Path to saved workload file
        episode_id: Episode identifier
        methods: List of methods to evaluate ('trained', 'ecmp', 'random', etc.)
        prev_queuing: Previous queuing delays for topk-queuing baseline
        prev_selected: Dict mapping method -> previous flow selection (for disturbance)
        temperature: Temperature for trained policy sampling
        save_episode: Whether to save episode outputs (default: False)
        
    Returns:
        Dictionary mapping method -> results
    """
    demand_vector = generate_demand_vector(workload_df)
    total_demand = np.sum(demand_vector)
    
    # Extract features for trained policy
    features = trainer.feature_extractor.extract_features(demand_vector)
    features_tensor = torch.FloatTensor(features).unsqueeze(0).to(trainer.device)
    
    # Get top-k queuing selection for overlap comparison
    if prev_queuing is not None:
        topk_q_flows = np.argsort(prev_queuing)[-trainer.config.k_critical:].tolist()
    else:
        topk_q_flows = []
    
    # Get top-k demand selection for overlap comparison
    topk_d_flows = np.argsort(demand_vector)[-trainer.config.k_critical:].tolist()
    
    if prev_selected is None:
        prev_selected = {}
    
    results = {}
    
    for method in methods:
        # Select flows based on method
        if method == 'trained':
            selected_flows = select_trained_policy_flows(
                trainer, features_tensor, temperature
            )
        else:
            selected_flows = select_baseline_flows(
                baseline_type=method,
                k=trainer.config.k_critical,
                demand_vector=demand_vector,
                prev_queuing=prev_queuing,
                num_flows=trainer.config.num_flows,
            )
        
        # Create output directory - only if saving this episode
        if save_episode:
            episode_dir = trainer.output_dir / f"baseline_{method}" / f"ep_{episode_id:04d}"
            episode_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Use temp directory, will be cleaned up
            episode_dir = trainer.output_dir / "temp_episode"
            episode_dir.mkdir(parents=True, exist_ok=True)
        
        # Run CloudSim episode
        episode_results = trainer.episode_runner.run_episode(
            workload_file=str(workload_path),
            critical_flows=selected_flows,
            output_dir=str(episode_dir),
            episode_id=episode_id,
        )
        
        trainer.episode_runner.cleanup_cloudsim_results()
        
        if not episode_results.get('success', False):
            # Clean up temp directory on failure
            if not save_episode:
                import shutil
                if episode_dir.exists():
                    shutil.rmtree(episode_dir)
            results[method] = {'success': False}
            continue
        
        # Compute reward BEFORE cleaning up temp directory
        episode_summary = episode_results.get('episode_summary', {})
        
        if trainer.config.use_dense_rewards:
            dense_result = compute_dense_reward_from_results(
                results=episode_results,
                window_size=trainer.config.dense_window_size,
                queuing_weight=trainer.config.queuing_weight,
                drop_penalty=trainer.config.drop_penalty,
                gamma=trainer.config.dense_gamma,
                critical_only=trainer.config.dense_critical_only,
            )
            reward = dense_result.get('total_reward', 0)
        else:
            reward = compute_reward(
                episode_summary=episode_summary,
                queuing_weight=trainer.config.queuing_weight,
                drop_penalty=trainer.config.drop_penalty,
            )
        
        # NOW clean up temp directory (after reading results)
        if not save_episode:
            import shutil
            if episode_dir.exists():
                shutil.rmtree(episode_dir)
        
        # Compute overlap with heuristics (only meaningful for trained policy)
        overlap_topk_q = compute_flow_overlap(selected_flows, topk_q_flows) if topk_q_flows else {}
        overlap_topk_d = compute_flow_overlap(selected_flows, topk_d_flows)
        
        # Compute traffic-weighted disturbance
        # Key metric: What % of total traffic is affected by critical flow selection?
        critical_demand = sum(demand_vector[f] for f in selected_flows if f < len(demand_vector))
        critical_traffic_pct = (critical_demand / max(total_demand, 1)) * 100
        
        # Selection churn: How many flows changed from last episode?
        prev_flows = set(prev_selected.get(method, []))
        curr_flows = set(selected_flows)
        churn = len(prev_flows.symmetric_difference(curr_flows))  # Flows added + flows removed
        
        results[method] = {
            'success': True,
            'reward': reward,
            'mean_queuing_ms': episode_summary.get('mean_queuing_delay_ms', 0),
            'max_queuing_ms': episode_summary.get('max_queuing_delay_ms', 0),
            'drop_rate': episode_summary.get('drop_rate', 0),
            'selected_flows': selected_flows,
            'flow_summary': episode_results.get('flow_summary'),
            # Overlap metrics
            'overlap_topk_queuing': overlap_topk_q.get('overlap_count', 0),
            'overlap_topk_demand': overlap_topk_d.get('overlap_count', 0),
            # Disturbance metrics
            'critical_traffic_pct': critical_traffic_pct,  # % of traffic affected
            'selection_churn': churn,  # How many flows changed status
        }
    
    return results


def run_fair_baseline_comparison(trainer, methods: list = None, num_episodes: int = 100,
                                  checkpoint_path: str = None, save_freq: int = 50) -> pd.DataFrame:
    """
    Run fair comparison where all methods are evaluated on the SAME workloads.
    
    This eliminates workload variance from the comparison, providing
    a true apples-to-apples evaluation.
    
    Args:
        trainer: CloudSimTrainer instance
        methods: List of methods to compare (default: all including 'trained' and 'all-critical')
        num_episodes: Number of episodes to run
        checkpoint_path: Path to trained policy checkpoint (required for 'trained')
        save_freq: Save episode outputs every N episodes (default: 50)
        
    Returns:
        DataFrame with per-episode results for all methods
    """
    from scipy import stats
    
    if methods is None:
        methods = ['trained', 'ecmp', 'random', 'topk-demand', 'topk-queuing', 'all-critical']
    
    # Load checkpoint if evaluating trained policy
    if 'trained' in methods and checkpoint_path:
        trainer.load_checkpoint(checkpoint_path)
        print(f"Loaded trained policy from: {checkpoint_path}")
    elif 'trained' in methods:
        print("Warning: No checkpoint provided, using current (possibly untrained) policy")
    
    print(f"\n{'='*70}")
    print(f"FAIR BASELINE COMPARISON")
    print(f"Methods: {', '.join(methods)}")
    print(f"Episodes: {num_episodes}")
    print(f"Same workload for all methods in each episode")
    print(f"Saving episode outputs every {save_freq} episodes")
    print(f"{'='*70}\n")
    
    # Store results per method - simplified metrics
    all_results = {method: {
        'rewards': [], 'queuing': [], 'max_queuing': [], 'drops': [],
        'overlap_topk_q': [], 'overlap_topk_d': [],
        'critical_traffic_pct': [], 'selection_churn': [],
    } for method in methods}
    
    # Track prev_queuing for topk-queuing baseline
    prev_queuing = {method: None for method in methods}
    # Track previous selections for churn calculation
    prev_selected = {method: [] for method in methods}
    
    for ep in range(num_episodes):
        # Determine if we should save this episode
        save_episode = (ep % save_freq == 0) or (ep == num_episodes - 1)
        
        # Generate ONE workload for this episode
        workload_df = generate_workload(
            num_packets=trainer.config.packets_per_episode,
            duration=trainer.config.episode_duration,
            seed=None,
            traffic_model=trainer.config.traffic_model,
        )
        
        # Save workload only for saved episodes
        workload_dir = trainer.output_dir / "shared_workloads"
        workload_dir.mkdir(parents=True, exist_ok=True)
        if save_episode:
            workload_path = workload_dir / f"workload_ep_{ep:04d}.csv"
        else:
            workload_path = workload_dir / "temp_workload.csv"
        save_workload(workload_df, str(workload_path))
        
        # Run all methods on this workload
        episode_results = run_single_episode_all_methods(
            trainer=trainer,
            workload_df=workload_df,
            workload_path=workload_path,
            episode_id=ep,
            methods=methods,
            prev_queuing=prev_queuing.get('topk-queuing'),
            prev_selected=prev_selected,
            temperature=trainer.config.min_temperature,
            save_episode=save_episode,
        )
        
        # Collect results and update tracking
        for method in methods:
            result = episode_results.get(method, {})
            if result.get('success', False):
                all_results[method]['rewards'].append(result['reward'])
                all_results[method]['queuing'].append(result['mean_queuing_ms'])
                all_results[method]['max_queuing'].append(result.get('max_queuing_ms', 0))
                all_results[method]['drops'].append(result['drop_rate'])
                all_results[method]['overlap_topk_q'].append(result.get('overlap_topk_queuing', 0))
                all_results[method]['overlap_topk_d'].append(result.get('overlap_topk_demand', 0))
                all_results[method]['critical_traffic_pct'].append(result.get('critical_traffic_pct', 0))
                all_results[method]['selection_churn'].append(result.get('selection_churn', 0))
                
                # Update previous selection for churn tracking
                prev_selected[method] = result.get('selected_flows', [])
                
                # Update prev_queuing from this method's results
                flow_summary = result.get('flow_summary')
                if flow_summary is not None and not flow_summary.empty:
                    pq = np.zeros(NUM_FLOWS)
                    for _, row in flow_summary.iterrows():
                        flow_id = int(row.get('flow_id', -1))
                        if 0 <= flow_id < NUM_FLOWS:
                            pq[flow_id] = row.get('mean_queuing_ms', 0)
                    prev_queuing[method] = pq
        
        # Progress update
        if (ep + 1) % 10 == 0 or ep == 0:
            status_parts = []
            for method in methods[:4]:  # Show first 4 methods
                if all_results[method]['rewards']:
                    r = all_results[method]['rewards'][-1]
                    status_parts.append(f"{method}={r:.1f}")
            print(f"[Ep {ep+1:3d}/{num_episodes}] {', '.join(status_parts)}")
    
    # Compute statistics
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}\n")
    
    summary_data = []
    trained_rewards = all_results.get('trained', {}).get('rewards', [])
    all_critical_rewards = all_results.get('all-critical', {}).get('rewards', [])
    
    # K values by method type
    k_by_method = {
        'ecmp': 0,
        'random': trainer.config.k_critical,
        'topk-demand': trainer.config.k_critical,
        'topk-queuing': trainer.config.k_critical,
        'trained': trainer.config.k_critical,
        'all-critical': trainer.config.num_flows,
    }
    
    for method in methods:
        rewards = all_results[method]['rewards']
        queuing = all_results[method]['queuing']
        
        if not rewards:
            print(f"{method}: No successful episodes")
            continue
        
        k_val = k_by_method.get(method, trainer.config.k_critical)
        
        row = {
            'Method': method.upper(),
            'K': k_val,
            'Episodes': len(rewards),
            'Mean Reward': np.mean(rewards),
            'Std Reward': np.std(rewards),
            'Median Reward': np.median(rewards),
            'Mean Queuing (ms)': np.mean(queuing),
        }
        
        # Traffic affected (only meaningful for K > 0)
        if all_results[method]['critical_traffic_pct'] and k_val > 0:
            row['Traffic Affected %'] = np.mean(all_results[method]['critical_traffic_pct'])
        
        # Selection churn (average flows changed per episode)
        if all_results[method]['selection_churn']:
            row['Avg Churn/Ep'] = np.mean(all_results[method]['selection_churn'])
        
        # Overlap with top-k heuristics (for trained policy)
        if method == 'trained' and all_results[method]['overlap_topk_q']:
            row['Overlap TopK-Q'] = f"{np.mean(all_results[method]['overlap_topk_q']):.1f}/{k_val}"
            row['Overlap TopK-D'] = f"{np.mean(all_results[method]['overlap_topk_d']):.1f}/{k_val}"
        
        # Compare to trained policy
        if method != 'trained' and trained_rewards:
            n = min(len(trained_rewards), len(rewards))
            t_stat, p_value = stats.ttest_rel(trained_rewards[:n], rewards[:n])
            improvement = (np.mean(trained_rewards) - np.mean(rewards)) / abs(np.mean(rewards)) * 100
            
            row['vs Trained (%)'] = improvement
            row['p-value'] = p_value
            row['Significant'] = 'YES' if p_value < 0.05 else 'no'
            row['Trained Wins'] = sum(1 for t, b in zip(trained_rewards[:n], rewards[:n]) if t > b)
        
        # Compare to all-critical (theoretical best)
        if method != 'all-critical' and all_critical_rewards:
            n = min(len(all_critical_rewards), len(rewards))
            improvement_vs_all = (np.mean(rewards) - np.mean(all_critical_rewards)) / abs(np.mean(all_critical_rewards)) * 100
            row['vs All-Crit (%)'] = improvement_vs_all
            # How much of the gap to all-critical does this method close?
            ecmp_rewards = all_results.get('ecmp', {}).get('rewards', [])
            if ecmp_rewards:
                ecmp_mean = np.mean(ecmp_rewards)
                all_crit_mean = np.mean(all_critical_rewards)
                method_mean = np.mean(rewards)
                if ecmp_mean != all_crit_mean:
                    gap_closed = (ecmp_mean - method_mean) / (ecmp_mean - all_crit_mean) * 100
                    row['Gap Closed %'] = gap_closed
        
        summary_data.append(row)
        
        # Print per-method summary
        print(f"{method.upper():15s}: R={np.mean(rewards):8.2f} ± {np.std(rewards):.2f}, "
              f"Q={np.mean(queuing):7.1f}ms  K={k_val:3d}", end="")
        if method != 'trained' and trained_rewards:
            print(f"  | vs trained: {row.get('vs Trained (%)', 0):+.1f}%", end="")
        if method != 'all-critical' and all_critical_rewards:
            print(f"  | gap closed: {row.get('Gap Closed %', 0):.0f}%", end="")
        print()
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Create detailed per-episode DataFrame
    detail_data = []
    for ep in range(num_episodes):
        row = {'episode': ep}
        for method in methods:
            if ep < len(all_results[method]['rewards']):
                row[f'{method}_reward'] = all_results[method]['rewards'][ep]
                row[f'{method}_queuing'] = all_results[method]['queuing'][ep]
                if method == 'trained':
                    row[f'{method}_overlap_topk_q'] = all_results[method]['overlap_topk_q'][ep]
                    row[f'{method}_overlap_topk_d'] = all_results[method]['overlap_topk_d'][ep]
        detail_data.append(row)
    
    detail_df = pd.DataFrame(detail_data)
    
    # Print summary table
    print(f"\n{'='*70}")
    print("COMPARISON TABLE")
    print(f"{'='*70}")
    print(summary_df.to_string(index=False))
    
    # Zhang et al. analysis: Is K-selection worth it?
    print(f"\n{'='*70}")
    print("ZHANG ET AL. ANALYSIS: Is K-selection worth the constraint?")
    print(f"{'='*70}")
    
    if all_critical_rewards and trained_rewards:
        all_crit_mean = np.mean(all_critical_rewards)
        trained_mean = np.mean(trained_rewards)
        ecmp_mean = np.mean(all_results.get('ecmp', {}).get('rewards', [0]))
        
        # Performance retention
        if ecmp_mean != all_crit_mean:
            trained_retention = (ecmp_mean - trained_mean) / (ecmp_mean - all_crit_mean) * 100
            print(f"\n  ECMP (K=0):           R={ecmp_mean:8.2f} (baseline, no rerouting)")
            print(f"  All-Critical (K=240): R={all_crit_mean:8.2f} (theoretical best, full TE)")
            print(f"  Trained (K=12):       R={trained_mean:8.2f}")
            print(f"\n  Performance retained: {trained_retention:.1f}% of full TE improvement")
            print(f"  Network disturbance:  {12/240*100:.1f}% of flows rerouted (12/240)")
            
            if trained_retention > 80:
                print(f"\n  ✓ VALIDATES Zhang et al.: {trained_retention:.0f}% performance with only 5% disturbance!")
            elif trained_retention > 50:
                print(f"\n  ~ PARTIAL: {trained_retention:.0f}% performance, may need more training or larger K")
            else:
                print(f"\n  ✗ K=12 may be insufficient for latency optimization")
    
    # Head-to-head analysis
    if 'trained' in methods and len(trained_rewards) > 0:
        print(f"\n{'='*70}")
        print("HEAD-TO-HEAD ANALYSIS")
        print(f"{'='*70}")
        
        for method in methods:
            if method == 'trained':
                continue
            
            rewards = all_results[method]['rewards']
            n = min(len(trained_rewards), len(rewards))
            
            wins = sum(1 for t, b in zip(trained_rewards[:n], rewards[:n]) if t > b)
            losses = sum(1 for t, b in zip(trained_rewards[:n], rewards[:n]) if t < b)
            ties = n - wins - losses
            
            print(f"  Trained vs {method:15s}: wins {wins:3d}, loses {losses:3d}, ties {ties:3d} "
                  f"({100*wins/n:.0f}% win rate)")
    
    # Overlap analysis for trained policy
    if 'trained' in methods and all_results['trained']['overlap_topk_q']:
        print(f"\n{'='*70}")
        print("TRAINED POLICY OVERLAP ANALYSIS")
        print(f"{'='*70}")
        overlap_q = all_results['trained']['overlap_topk_q']
        overlap_d = all_results['trained']['overlap_topk_d']
        print(f"  Overlap with Top-K Queuing: {np.mean(overlap_q):.1f}/12 "
              f"(min={min(overlap_q)}, max={max(overlap_q)})")
        print(f"  Overlap with Top-K Demand:  {np.mean(overlap_d):.1f}/12 "
              f"(min={min(overlap_d)}, max={max(overlap_d)})")
        
        if np.mean(overlap_q) > 10:
            print(f"\n  ⚠️ High overlap with Top-K Queuing - policy may be copying heuristic")
        elif np.mean(overlap_q) < 6:
            print(f"\n  ✓ Low overlap - policy found different strategy than Top-K Queuing")
        else:
            print(f"\n  ~ Moderate overlap - policy partially diverged from heuristic")
    
    return summary_df, detail_df


def run_baseline_comparison(trainer, baselines: list = None, episodes_per_baseline: int = 100,
                            trained_rewards: list = None) -> pd.DataFrame:
    """
    Legacy function - runs baselines independently (not on same workloads).
    
    For fair comparison, use run_fair_baseline_comparison() instead.
    """
    from scipy import stats
    
    print("\nWARNING: This runs baselines on different workloads.")
    print("For fair comparison, use --baseline all with --checkpoint\n")
    
    if baselines is None:
        baselines = ['ecmp', 'random', 'topk-demand', 'topk-queuing']
    
    all_results = []
    
    for baseline in baselines:
        results = evaluate_baseline_legacy(
            trainer=trainer,
            baseline_type=baseline,
            num_episodes=episodes_per_baseline,
            use_dense_rewards=trainer.config.use_dense_rewards,
        )
        all_results.append(results)
    
    # Build comparison table
    comparison_data = []
    
    for results in all_results:
        row = {
            'Method': results['baseline'],
            'Mean Reward': results['mean_reward'],
            'Std Reward': results['std_reward'],
            'Mean Queuing (ms)': results['mean_queuing_ms'],
        }
        
        if trained_rewards is not None:
            t_stat, p_value = stats.ttest_ind(trained_rewards, results['rewards'])
            improvement = (np.mean(trained_rewards) - results['mean_reward']) / abs(results['mean_reward']) * 100
            row['vs Trained (%)'] = improvement
            row['p-value'] = p_value
            row['Significant'] = 'Yes' if p_value < 0.05 else 'No'
        
        comparison_data.append(row)
    
    if trained_rewards is not None:
        trained_row = {
            'Method': 'TRAINED POLICY',
            'Mean Reward': np.mean(trained_rewards),
            'Std Reward': np.std(trained_rewards),
            'Mean Queuing (ms)': np.nan,
            'vs Trained (%)': 0.0,
            'p-value': 1.0,
            'Significant': '-',
        }
        comparison_data.append(trained_row)
    
    df = pd.DataFrame(comparison_data)
    
    print("\n" + "="*70)
    print("BASELINE COMPARISON SUMMARY")
    print("="*70)
    print(df.to_string(index=False))
    
    return df


def evaluate_baseline_legacy(trainer, baseline_type: str, num_episodes: int = 100,
                              use_dense_rewards: bool = True) -> dict:
    """Legacy baseline evaluation (independent workloads)."""
    
    print(f"\n{'='*70}")
    print(f"BASELINE EVALUATION: {baseline_type.upper()}")
    print(f"{'='*70}")
    
    rewards = []
    queuing_delays = []
    drop_rates = []
    prev_queuing = None
    
    for ep in range(num_episodes):
        workload_df = generate_workload(
            num_packets=trainer.config.packets_per_episode,
            duration=trainer.config.episode_duration,
            seed=None,
            traffic_model=trainer.config.traffic_model,
        )
        
        episode_dir = trainer.output_dir / f"baseline_{baseline_type}" / f"ep_{ep:04d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        workload_path = episode_dir / "workload.csv"
        save_workload(workload_df, str(workload_path))
        
        demand_vector = generate_demand_vector(workload_df)
        
        selected_flows = select_baseline_flows(
            baseline_type=baseline_type,
            k=trainer.config.k_critical,
            demand_vector=demand_vector,
            prev_queuing=prev_queuing,
            num_flows=trainer.config.num_flows,
        )
        
        results = trainer.episode_runner.run_episode(
            workload_file=str(workload_path),
            critical_flows=selected_flows,
            output_dir=str(episode_dir),
            episode_id=ep,
        )
        
        trainer.episode_runner.cleanup_cloudsim_results()
        
        if not results.get('success', False):
            continue
        
        episode_summary = results.get('episode_summary', {})
        
        if use_dense_rewards:
            dense_result = compute_dense_reward_from_results(
                results=results,
                window_size=trainer.config.dense_window_size,
                queuing_weight=trainer.config.queuing_weight,
                drop_penalty=trainer.config.drop_penalty,
                gamma=trainer.config.dense_gamma,
                critical_only=trainer.config.dense_critical_only,
            )
            reward = dense_result.get('total_reward', 0)
        else:
            reward = compute_reward(
                episode_summary=episode_summary,
                queuing_weight=trainer.config.queuing_weight,
                drop_penalty=trainer.config.drop_penalty,
            )
        
        rewards.append(reward)
        queuing_delays.append(episode_summary.get('mean_queuing_delay_ms', 0))
        drop_rates.append(episode_summary.get('drop_rate', 0))
        
        flow_summary = results.get('flow_summary')
        if flow_summary is not None and not flow_summary.empty:
            prev_queuing = np.zeros(NUM_FLOWS)
            for _, row in flow_summary.iterrows():
                flow_id = int(row.get('flow_id', -1))
                if 0 <= flow_id < NUM_FLOWS:
                    prev_queuing[flow_id] = row.get('mean_queuing_ms', 0)
        
        if (ep + 1) % 10 == 0:
            print(f"  [{baseline_type}] Episode {ep+1}/{num_episodes}: "
                  f"R={reward:.4f}, Q={queuing_delays[-1]:.1f}ms")
    
    return {
        'baseline': baseline_type,
        'num_episodes': len(rewards),
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'median_reward': np.median(rewards),
        'mean_queuing_ms': np.mean(queuing_delays),
        'std_queuing_ms': np.std(queuing_delays),
        'mean_drop_rate': np.mean(drop_rates),
        'rewards': rewards,
    }


# =============================================================================
# Main Trainer
# =============================================================================

class CloudSimTrainer:
    """
    Main trainer for CloudSim-in-the-loop RL.
    
    Supports both PPO and REINFORCE algorithms.
    Supports sparse and dense reward modes.
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
        
        # Initialize networks based on algorithm choice
        if config.algorithm == 'ppo':
            self._init_ppo(config)
        else:
            self._init_reinforce(config)
        
        self.feature_extractor = FeatureExtractor(random_cold_start=True)
        
        self.episode_runner = EpisodeRunner(
            cloudsim_dir=config.cloudsim_dir,
            timeout=120,
            verbose=True,
        )
        
        self.logger = MetricsLogger(
            str(self.output_dir), 
            config.algorithm,
            config.use_dense_rewards
        )
        
        # Temperature schedule
        self.temperature = config.temperature_start
        
        # Best model tracking
        self.best_reward = float('-inf')
        
        # Episode counter
        self.episode = 0

        # Fixed workload support (reuse within batch for fair comparison)
        self._cached_workload = None
        self._cached_workload_batch = -1
        
        # Save config
        config_path = self.output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2, default=str)
        
        print("CloudSimTrainer initialized")
        print(f"  Algorithm: {config.algorithm.upper()}")
        print(f"  Rewards: {'DENSE' if config.use_dense_rewards else 'SPARSE'}")
        if config.use_dense_rewards:
            print(f"    Window size: {config.dense_window_size}s")
            print(f"    Discount (gamma): {config.dense_gamma}")
            print(f"    Critical only: {config.dense_critical_only}")
        print(f"  Temperature: {config.temperature_start} → {config.min_temperature} (min floor)")
        print(f"  Topology: Fat-Tree k=4 ({NUM_HOSTS} hosts, {NUM_FLOWS} flows)")
        print(f"  Device: {self.device}")
        print(f"  Output dir: {self.output_dir}")
        print(f"  Policy parameters: {sum(p.numel() for p in self.policy.parameters()):,}")
        if config.algorithm == 'ppo':
            print(f"  Value net parameters: {sum(p.numel() for p in self.value_net.parameters()):,}")
    
    def _init_ppo(self, config: TrainingConfig) -> None:
        """Initialize PPO networks and trainer."""
        self.policy, self.value_net = create_ppo_networks(
            num_flows=config.num_flows,
            num_features=config.num_features,
            hidden_dim=config.hidden_dims[0],
            num_hidden_layers=len(config.hidden_dims),
            device=self.device,
        )
        
        self.trainer = PPOTrainer(
            policy=self.policy,
            value_net=self.value_net,
            lr_policy=config.learning_rate,
            lr_value=config.lr_value,
            batch_size=config.batch_size,
            ppo_epochs=config.ppo_epochs,
            clip_epsilon=config.clip_epsilon,
            value_loss_coef=config.value_loss_coef,
            entropy_coef=config.entropy_weight,
            max_grad_norm=config.max_grad_norm,
            gae_lambda=config.gae_lambda,
            normalize_advantages=config.normalize_advantages,
            target_kl=config.target_kl,
        )
    
    def _init_reinforce(self, config: TrainingConfig) -> None:
        """Initialize REINFORCE network and trainer."""
        self.policy = PolicyNetwork(
            num_flows=config.num_flows,
            num_features=config.num_features,
            hidden_dim=config.hidden_dims[0],
            num_hidden_layers=len(config.hidden_dims),
        ).to(self.device)
        
        self.value_net = None
        
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
    
    def save_checkpoint(self, path: str, is_best: bool = False) -> None:
        """Save training checkpoint."""
        checkpoint = {
            'episode': self.episode,
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'baseline': self.trainer.get_baseline(),
            'temperature': self.temperature,
            'best_reward': self.best_reward,
            'config': self.config.to_dict(),
            'algorithm': self.config.algorithm,
            'use_dense_rewards': self.config.use_dense_rewards,
        }
        
        if self.config.algorithm == 'ppo':
            checkpoint['value_net_state_dict'] = self.value_net.state_dict()
            checkpoint['value_optimizer_state_dict'] = self.trainer.value_optimizer.state_dict()
        
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        # weights_only=False needed for PyTorch 2.6+ (changed default)
        # Safe here since we're loading our own checkpoint files
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        saved_algo = checkpoint.get('algorithm', 'reinforce')
        if saved_algo != self.config.algorithm:
            print(f"Warning: Checkpoint was saved with {saved_algo}, but current config uses {self.config.algorithm}")
            print(f"Loading policy weights only...")
        
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.temperature = checkpoint['temperature']
        self.best_reward = checkpoint['best_reward']
        self.episode = checkpoint['episode']
        
        if self.config.algorithm == 'ppo' and 'value_net_state_dict' in checkpoint:
            self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
            self.trainer.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        
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
            Dictionary with episode results including reward (sparse or dense)
        """
        if existing_workload:
            workload_path = Path(self.config.cloudsim_dir) / existing_workload
            episode_dir = f"episodes/ep_{episode_id:04d}"
            workload_df = load_workload(str(workload_path))
            workload_file = existing_workload
            should_save = True
        else:
            save_freq = self.config.save_episode_freq
            should_save = save_freq > 0 and (episode_id % save_freq == 0)
            
            if should_save:
                episode_dir = f"episodes/ep_{episode_id:04d}"
            else:
                episode_dir = "episodes/temp"
            
            current_batch = episode_id // self.config.batch_size
            
            if current_batch != self._cached_workload_batch:
                self._cached_workload = generate_workload(
                    num_packets=self.config.packets_per_episode,
                    duration=self.config.episode_duration,
                    seed=None,
                    traffic_model=self.config.traffic_model,
                )
                self._cached_workload_batch = current_batch
                print(f"[Trainer] Batch {current_batch}: Generated new fixed workload")
            else:
                ep_in_batch = episode_id % self.config.batch_size
                print(f"[Trainer] Batch {current_batch}: Reusing workload (ep {ep_in_batch + 1}/{self.config.batch_size})")
            
            workload_df = self._cached_workload.copy()
            
            workload_file = f"{episode_dir}/workload.csv"
            workload_path = self.output_dir / workload_file
            workload_path.parent.mkdir(parents=True, exist_ok=True)
            save_workload(workload_df, str(workload_path))
        
        # Extract features
        demand_vector = generate_demand_vector(workload_df)
        features = self.feature_extractor.extract_features(demand_vector)
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        # Get policy action
        logits = self.policy(features_tensor)
        selected_flows, log_prob = self.policy.sample_action(
            logits, 
            k=self.config.k_critical,
            temperature=self.temperature,
        )
        
        # Run CloudSim episode
        output_dir = str(self.output_dir / episode_dir)
        results = self.episode_runner.run_episode(
            workload_file=str(workload_path),
            critical_flows=selected_flows,
            output_dir=output_dir,
            episode_id=episode_id,
        )
        
        self.episode_runner.cleanup_cloudsim_results()
        
        if not results.get('success', False):
            print(f"  Episode {episode_id} FAILED: {results.get('error', 'Unknown error')}")
            if not should_save:
                temp_dir = self.output_dir / "episodes" / "temp"
                if temp_dir.exists():
                    import shutil
                    shutil.rmtree(temp_dir)
            return {'success': False, 'reward': -10.0}
        
        # Compute reward (sparse or dense)
        episode_summary = results.get('episode_summary', {})
        dense_info = {}
        
        if self.config.use_dense_rewards:
            dense_result = compute_dense_reward_from_results(
                results=results,
                window_size=self.config.dense_window_size,
                queuing_weight=self.config.queuing_weight,
                drop_penalty=self.config.drop_penalty,
                gamma=self.config.dense_gamma,
                critical_only=self.config.dense_critical_only,
            )
            
            reward = dense_result.get('total_reward', 0)
            
            dense_info = {
                'num_windows': dense_result.get('num_windows', 0),
                'critical_packets': dense_result.get('critical_packets', 0),
                'dense_reward_undiscounted': dense_result.get('total_reward_undiscounted', 0),
            }
            
            if should_save or episode_id % 50 == 0:
                sparse_reward = compute_reward(
                    episode_summary=episode_summary,
                    queuing_weight=self.config.queuing_weight,
                    drop_penalty=self.config.drop_penalty,
                )
                print(f"    [Dense] windows={dense_info['num_windows']}, "
                      f"crit_packets={dense_info['critical_packets']}, "
                      f"R_dense={reward:.4f}, R_sparse={sparse_reward:.4f}")
        else:
            reward = compute_reward(
                episode_summary=episode_summary,
                queuing_weight=self.config.queuing_weight,
                drop_penalty=self.config.drop_penalty,
            )
        
        # Update feature extractor
        flow_summary = results.get('flow_summary')
        if flow_summary is not None and not flow_summary.empty:
            self.feature_extractor.update_historical_features(flow_summary)
        
        # Clean up temp folder
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
            **dense_info,
        }
    
    def train(self, start_episode: int = 0, end_episode: int = None) -> None:
        """Run training loop with batch updates."""
        if end_episode is None:
            end_episode = self.config.num_episodes
        
        algo_name = self.config.algorithm.upper()
        reward_type = "DENSE" if self.config.use_dense_rewards else "SPARSE"
        print(f"Starting {algo_name} training ({reward_type} rewards) from episode {start_episode} to {end_episode}")
        print(f"  Batch size: {self.config.batch_size}")
        if self.config.algorithm == 'ppo':
            print(f"  PPO epochs per batch: {self.config.ppo_epochs}")
            print(f"  Clip epsilon: {self.config.clip_epsilon}")
        else:
            print(f"  Updates per batch: {self.config.num_updates_per_batch}")
        if self.config.use_dense_rewards:
            print(f"  Dense window size: {self.config.dense_window_size}s")
        print("=" * 70)
        
        for episode_id in range(start_episode, end_episode):
            self.episode = episode_id
            
            results = self.run_episode(episode_id)
            
            if not results.get('success', False):
                continue
            
            reward = results['reward']
            log_prob = results['log_prob']
            features_tensor = results['features_tensor']
            selected_flows = results['selected_flows']
            episode_summary = results.get('episode_summary', {})
            
            self.trainer.store_episode(
                features=features_tensor,
                selected_flows=selected_flows,
                log_prob=log_prob,
                reward=reward,
                temperature=self.temperature,
            )
            
            update_metrics = {}
            if self.trainer.should_update():
                update_metrics = self.trainer.update()
                
                if self.config.algorithm == 'ppo':
                    print(f"    [{algo_name} Update] "
                          f"policy_loss={update_metrics.get('policy_loss', 0):.4f} "
                          f"value_loss={update_metrics.get('value_loss', 0):.4f} "
                          f"entropy={update_metrics.get('entropy', 0):.4f} "
                          f"kl={update_metrics.get('kl_divergence', 0):.4f} "
                          f"epochs={update_metrics.get('num_epochs', 0)}")
                else:
                    print(f"    [REINFORCE Update] loss={update_metrics.get('loss', 0):.4f} "
                          f"batch_R={update_metrics.get('batch_reward_mean', 0):.4f}")
            
            self.temperature = max(
                self.config.min_temperature,
                self.temperature * self.config.temperature_decay
            )
            
            metrics = {
                'reward': reward,
                'mean_queuing_ms': episode_summary.get('mean_queuing_ms', 0),
                'drop_rate': episode_summary.get('drop_rate', 0),
                'loss': update_metrics.get('loss', 0),
                'policy_loss': update_metrics.get('policy_loss', 0),
                'value_loss': update_metrics.get('value_loss', 0),
                'entropy': update_metrics.get('entropy', 0),
                'kl_divergence': update_metrics.get('kl_divergence', 0),
                'baseline': self.trainer.get_baseline(),
                'temperature': self.temperature,
                'wall_time_ms': results.get('wall_time_ms', 0),
                'total_packets': episode_summary.get('total_packets', 0),
                'num_flows_active': episode_summary.get('num_flows_active', 0),
                'num_windows': results.get('num_windows', 0),
                'critical_packets': results.get('critical_packets', 0),
                'dense_reward_undiscounted': results.get('dense_reward_undiscounted', 0),
            }
            self.logger.log(episode_id, metrics)
            
            stats = self.logger.get_recent_stats(10)
            print(f"[Ep {episode_id:4d}] R={reward:.4f} Q={metrics['mean_queuing_ms']:.1f}ms "
                  f"D={metrics['drop_rate']:.4f} T={self.temperature:.3f} | "
                  f"Avg10: R={stats.get('mean_reward', 0):.4f}")
            
            if reward > self.best_reward:
                self.best_reward = reward
                self.save_checkpoint(
                    str(self.checkpoint_dir / "best_model.pt"),
                    is_best=True
                )
                print(f"    ★ New best reward: {reward:.6f}")
            
            if (episode_id + 1) % self.config.checkpoint_freq == 0:
                self.save_checkpoint(
                    str(self.checkpoint_dir / f"checkpoint_ep{episode_id:04d}.pt")
                )
        
        self.save_checkpoint(str(self.checkpoint_dir / "final_model.pt"))
        
        print("=" * 70)
        print("Training complete!")
        print(f"  Algorithm: {self.config.algorithm.upper()}")
        print(f"  Rewards: {'DENSE' if self.config.use_dense_rewards else 'SPARSE'}")
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
    """Evaluate a trained policy."""
    rewards = []
    queuing_delays = []
    drop_rates = []
    
    trainer.policy.eval()
    if trainer.value_net is not None:
        trainer.value_net.eval()
    
    original_temp = trainer.temperature
    trainer.temperature = 0.1
    
    for ep in range(num_episodes):
        results = trainer.run_episode(episode_id=10000 + ep)
        
        if results.get('success', False):
            rewards.append(results['reward'])
            summary = results.get('episode_summary', {})
            queuing_delays.append(summary.get('mean_queuing_ms', 0))
            drop_rates.append(summary.get('drop_rate', 0))
    
    trainer.temperature = original_temp
    
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
    
    parser.add_argument("--cloudsim-dir", type=str, required=True,
                       help="Path to CloudSimSDN project directory")
    
    parser.add_argument("--algorithm", type=str, default='ppo',
                       choices=['ppo', 'reinforce'],
                       help="Training algorithm (default: ppo)")
    
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--packets", type=int, default=300)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--k-critical", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--traffic-model", type=str, default='hotspot',
                       choices=['uniform', 'hotspot', 'gravity', 'skewed'])
    parser.add_argument("--save-episode-freq", type=int, default=50)
    
    # PPO-specific
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--lr-value", type=float, default=1e-3)
    
    # Dense rewards
    parser.add_argument("--dense-rewards", action="store_true",
                       help="Use dense (windowed) rewards instead of sparse")
    parser.add_argument("--window-size", type=float, default=1.0,
                       help="Window size for dense rewards in seconds")
    parser.add_argument("--dense-gamma", type=float, default=0.99,
                       help="Discount factor for dense rewards")
    parser.add_argument("--dense-all-packets", action="store_true",
                       help="Use all packets for dense rewards (default: critical only)")
    
    # Temperature control
    parser.add_argument("--min-temperature", type=float, default=0.1,
                       help="Minimum temperature floor (default: 0.1, use 0.15 to prevent over-exploitation)")
    
    parser.add_argument("--output-dir", type=str, default="training_outputs")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval-only", action="store_true")
    
    # Baseline evaluation
    parser.add_argument("--baseline", type=str, default=None,
                       choices=['ecmp', 'random', 'topk-demand', 'topk-queuing', 'all-critical', 'all'],
                       help="Run baseline evaluation instead of training")
    parser.add_argument("--baseline-episodes", type=int, default=100,
                       help="Number of episodes per baseline evaluation")
    parser.add_argument("--compare-trained", type=str, default=None,
                       help="Path to training_metrics.csv to compare baselines against (legacy mode)")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to trained policy checkpoint for fair comparison")
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        algorithm=args.algorithm,
        num_episodes=args.episodes,
        packets_per_episode=args.packets,
        episode_duration=args.duration,
        k_critical=args.k_critical,
        traffic_model=args.traffic_model,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        ppo_epochs=args.ppo_epochs,
        clip_epsilon=args.clip_epsilon,
        lr_value=args.lr_value,
        save_episode_freq=args.save_episode_freq,
        use_dense_rewards=args.dense_rewards,
        dense_window_size=args.window_size,
        dense_gamma=args.dense_gamma,
        dense_critical_only=not args.dense_all_packets,
        min_temperature=args.min_temperature,
        cloudsim_dir=args.cloudsim_dir,
        output_dir=args.output_dir,
    )
    
    trainer = CloudSimTrainer(config)
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
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
    
    # Baseline evaluation mode
    if args.baseline:
        # Determine which methods to evaluate
        if args.baseline == 'all':
            methods = ['ecmp', 'random', 'topk-demand', 'topk-queuing', 'all-critical']
            if args.checkpoint:
                methods = ['trained'] + methods  # Add trained policy if checkpoint provided
        else:
            methods = [args.baseline]
            if args.checkpoint:
                methods = ['trained'] + methods
        
        # Use fair comparison (same workloads) if checkpoint provided
        if args.checkpoint:
            summary_df, detail_df = run_fair_baseline_comparison(
                trainer=trainer,
                methods=methods,
                num_episodes=args.baseline_episodes,
                checkpoint_path=args.checkpoint,
            )
            
            # Save results
            summary_path = trainer.output_dir / "baseline_comparison_summary.csv"
            detail_path = trainer.output_dir / "baseline_comparison_detail.csv"
            summary_df.to_csv(summary_path, index=False)
            detail_df.to_csv(detail_path, index=False)
            print(f"\nSummary saved to: {summary_path}")
            print(f"Details saved to: {detail_path}")
        
        else:
            # Legacy mode: different workloads, compare to historical metrics
            trained_rewards = None
            if args.compare_trained:
                try:
                    trained_df = pd.read_csv(args.compare_trained)
                    n_compare = min(100, len(trained_df))
                    trained_rewards = trained_df['reward'].tail(n_compare).tolist()
                    print(f"Loaded {n_compare} episodes from trained policy for comparison")
                except Exception as e:
                    print(f"Warning: Could not load trained metrics: {e}")
            
            comparison_df = run_baseline_comparison(
                trainer=trainer,
                baselines=methods,
                episodes_per_baseline=args.baseline_episodes,
                trained_rewards=trained_rewards,
            )
            
            results_path = trainer.output_dir / "baseline_comparison.csv"
            comparison_df.to_csv(results_path, index=False)
            print(f"\nResults saved to: {results_path}")
        
        return
    
    start_ep = trainer.episode if args.resume else 0
    trainer.train(start_episode=start_ep, end_episode=args.episodes)


if __name__ == "__main__":
    main()