"""
Pure RL Trainer for Abilene Topology

KEY DIFFERENCE: No supervised pre-training on Top-K!

The policy learns ONLY from the utilization reward signal.
This lets it potentially discover strategies that Top-K misses.

Training approach:
1. Generate random traffic
2. Policy selects K flows
3. LP solver computes resulting utilization
4. Reward = improvement over baseline
5. Policy gradient update
6. Repeat 10,000+ times
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple
from dataclasses import dataclass
import time

from abilene_topology import AbileneToplogy
from abilene_lp_solver import AbileneLPSolver
from abilene_traffic_gen import AbileneTrafficGenerator


class PointwisePolicy(nn.Module):
    """
    Pointwise scoring network.
    
    For each flow, predicts a score indicating how valuable it is to optimize.
    Selection: take top-K flows by score.
    """
    
    def __init__(self, num_flows: int, hidden_dim: int = 64):
        super().__init__()
        self.num_flows = num_flows
        
        # Per-flow feature encoder
        # Input per flow: [demand, num_paths, path_length, bottleneck_capacity]
        self.flow_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Global context: aggregate all flow info
        self.global_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Score head: combines local and global info
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, flow_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            flow_features: [batch, num_flows, 4]
            
        Returns:
            scores: [batch, num_flows]
        """
        batch_size = flow_features.shape[0]
        
        # Encode each flow locally
        local_encoded = self.flow_encoder(flow_features)  # [batch, num_flows, hidden]
        
        # Global context: mean pooling over flows
        global_context = local_encoded.mean(dim=1, keepdim=True)  # [batch, 1, hidden]
        global_context = self.global_encoder(global_context)
        global_context = global_context.expand(-1, self.num_flows, -1)  # [batch, num_flows, hidden]
        
        # Combine local and global
        combined = torch.cat([local_encoded, global_context], dim=-1)  # [batch, num_flows, hidden*2]
        
        # Score each flow
        scores = self.score_head(combined).squeeze(-1)  # [batch, num_flows]
        
        return scores
    
    def select_flows(self, flow_features: np.ndarray, k: int,
                    deterministic: bool = False,
                    temperature: float = 1.0,
                    mask: np.ndarray = None) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
        """
        Select k flows.
        
        Args:
            flow_features: [num_flows, 4]
            k: Number of flows to select
            deterministic: If True, take top-k by score
            temperature: Sampling temperature
            mask: Optional mask (1 = eligible, 0 = ineligible)
            
        Returns:
            selected: Flow indices
            log_prob: Log probability of selection
            entropy: Distribution entropy
        """
        x = torch.FloatTensor(flow_features).unsqueeze(0)  # [1, num_flows, 4]
        scores = self.forward(x).squeeze(0)  # [num_flows]
        
        # Apply mask if provided (set ineligible flows to -inf)
        if mask is not None:
            mask_tensor = torch.FloatTensor(mask)
            scores = scores + (1 - mask_tensor) * (-1e9)
        
        if deterministic:
            selected = torch.topk(scores, k).indices.tolist()
            log_prob = torch.tensor(0.0)
            entropy = torch.tensor(0.0)
        else:
            # Softmax with temperature
            probs = torch.softmax(scores / temperature, dim=0)
            
            # Handle numerical issues
            probs = probs.clamp(min=1e-10)
            probs = probs / probs.sum()
            
            # Sample without replacement
            try:
                selected = torch.multinomial(probs, k, replacement=False).tolist()
            except RuntimeError:
                # Fallback to top-k if sampling fails
                selected = torch.topk(scores, k).indices.tolist()
            
            # Log probability
            log_probs = torch.log_softmax(scores / temperature, dim=0)
            log_prob = log_probs[selected].sum()
            
            # Entropy
            entropy = -(probs * torch.log(probs + 1e-10)).sum()
        
        return selected, log_prob, entropy


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Core hyperparameters
    k_critical: int = 8
    hidden_dim: int = 64
    
    # Training
    total_iterations: int = 10000
    lr: float = 3e-4
    entropy_coef: float = 0.05      # Higher = more exploration
    grad_clip: float = 0.5
    temperature: float = 1.0        # Sampling temperature
    temperature_decay: float = 0.9995  # Anneal temperature over time
    min_temperature: float = 0.1
    
    # Evaluation
    eval_interval: int = 500
    eval_episodes: int = 50
    log_interval: int = 100


class AbileneRLTrainer:
    """
    Pure RL trainer for Abilene.
    
    NO supervised pre-training!
    Policy learns entirely from utilization reward.
    """
    
    def __init__(self, config: TrainingConfig = None, seed: int = 42):
        self.config = config or TrainingConfig()
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Components
        self.topo = AbileneToplogy()
        self.solver = AbileneLPSolver(self.topo)
        self.traffic_gen = AbileneTrafficGenerator(self.topo, seed=seed)
        
        # Policy
        self.policy = PointwisePolicy(
            num_flows=self.topo.num_flows,
            hidden_dim=self.config.hidden_dim
        )
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.config.lr)
        
        # Flow metadata
        self._compute_flow_metadata()
        
        # Tracking
        self.history = []
        self.best_performance = float('inf')
        self.current_temperature = self.config.temperature
        
    def _compute_flow_metadata(self):
        """Precompute static flow features."""
        self.num_paths = np.zeros(self.topo.num_flows)
        self.path_lengths = np.zeros(self.topo.num_flows)
        self.bottleneck_caps = np.zeros(self.topo.num_flows)
        self.is_multi_path = np.zeros(self.topo.num_flows)
        
        for flow_idx, (src, dst) in enumerate(self.topo.flow_pairs):
            paths = self.topo.get_all_paths(src, dst)
            self.num_paths[flow_idx] = len(paths)
            self.is_multi_path[flow_idx] = float(len(paths) > 1)
            
            if paths:
                self.path_lengths[flow_idx] = len(paths[0]) - 1  # Hop count
                self.bottleneck_caps[flow_idx] = self.topo.get_path_bottleneck(paths[0])
            else:
                self.path_lengths[flow_idx] = 0
                self.bottleneck_caps[flow_idx] = 0
    
    def prepare_features(self, tm: np.ndarray) -> np.ndarray:
        """
        Prepare per-flow features.
        
        Features: [normalized_demand, num_paths, path_length, bottleneck_cap]
        """
        flows = self.traffic_gen.tm_to_flow_vector(tm)
        
        max_flow = flows.max() if flows.max() > 0 else 1.0
        max_cap = self.bottleneck_caps.max() if self.bottleneck_caps.max() > 0 else 1.0
        
        features = np.stack([
            flows / max_flow,                           # Normalized demand
            self.num_paths / 4.0,                       # Normalized path count
            self.path_lengths / 5.0,                    # Normalized path length
            self.bottleneck_caps / max_cap,             # Normalized bottleneck
        ], axis=1)
        
        return features
    
    def get_top_k_indices(self, tm: np.ndarray) -> List[int]:
        """Get Top-K selection (for comparison only)."""
        flows = self.traffic_gen.tm_to_flow_vector(tm)
        masked = flows * self.is_multi_path
        return np.argsort(-masked)[:self.config.k_critical].tolist()
    
    def train_step(self) -> Dict:
        """
        Single training step.
        
        Returns metrics dict.
        """
        # Generate random traffic (diverse patterns)
        pattern = np.random.choice(['bimodal', 'gravity', 'hotspot', 'realistic'])
        if pattern == 'bimodal':
            tm = self.traffic_gen.generate_bimodal()
        elif pattern == 'gravity':
            tm = self.traffic_gen.generate_gravity()
        elif pattern == 'hotspot':
            tm = self.traffic_gen.generate_hotspot()
        else:
            tm = self.traffic_gen.generate_realistic()
        
        features = self.prepare_features(tm)
        
        # Baselines
        util_single = self.solver.solve_single_path(tm)
        util_top_k, top_k_flows = self.solver.solve_top_k(tm, self.config.k_critical)
        
        # Policy selection (with exploration)
        selected, log_prob, entropy = self.policy.select_flows(
            features,
            self.config.k_critical,
            deterministic=False,
            temperature=self.current_temperature,
            mask=self.is_multi_path  # Only select multi-path flows
        )
        
        # Evaluate policy's selection
        util_policy, _ = self.solver.solve(tm, selected, background_routing="single_path")
        
        # Reward: relative improvement over single-path baseline
        # Scale to roughly [-1, 1] range
        reward = (util_single - util_policy) / max(util_single, 0.01)
        
        # Advantage: compare to running average (not Top-K!)
        # This lets the policy discover its own optimum
        baseline = getattr(self, 'reward_baseline', reward)
        advantage = reward - baseline
        
        # Update baseline with exponential moving average
        self.reward_baseline = 0.99 * baseline + 0.01 * reward
        
        # Policy gradient loss
        policy_loss = -log_prob * advantage
        entropy_loss = -self.config.entropy_coef * entropy
        loss = policy_loss + entropy_loss
        
        # Gradient update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.grad_clip)
        self.optimizer.step()
        
        # Temperature decay
        self.current_temperature = max(
            self.config.min_temperature,
            self.current_temperature * self.config.temperature_decay
        )
        
        # Check if policy selected same as Top-K
        overlap = len(set(selected) & set(top_k_flows))
        
        return {
            'util_policy': util_policy,
            'util_top_k': util_top_k,
            'util_single': util_single,
            'reward': reward,
            'entropy': entropy.item(),
            'overlap': overlap,
            'temperature': self.current_temperature,
        }
    
    def train(self) -> List[Dict]:
        """
        Full training loop.
        """
        print(f"\n{'='*60}")
        print("PURE RL TRAINING (NO TOP-K PRE-TRAINING)")
        print(f"{'='*60}")
        print(f"Iterations: {self.config.total_iterations}")
        print(f"K critical: {self.config.k_critical}")
        print(f"Entropy coef: {self.config.entropy_coef}")
        print(f"Temperature: {self.config.temperature} -> {self.config.min_temperature}")
        
        start_time = time.time()
        
        for i in range(self.config.total_iterations):
            metrics = self.train_step()
            self.history.append(metrics)
            
            # Logging
            if (i + 1) % self.config.log_interval == 0:
                recent = self.history[-self.config.log_interval:]
                avg_policy = np.mean([m['util_policy'] for m in recent])
                avg_topk = np.mean([m['util_top_k'] for m in recent])
                avg_entropy = np.mean([m['entropy'] for m in recent])
                avg_overlap = np.mean([m['overlap'] for m in recent])
                temp = metrics['temperature']
                
                # Win rate: how often does policy beat or match Top-K?
                wins = sum(1 for m in recent if m['util_policy'] <= m['util_top_k'] * 1.01)
                win_rate = wins / len(recent) * 100
                
                status = "✓" if avg_policy <= avg_topk else "✗"
                print(f"  Iter {i+1:5d} | Policy: {avg_policy:.3f} | Top-K: {avg_topk:.3f} | "
                      f"Win: {win_rate:4.0f}% | Overlap: {avg_overlap:.1f}/{self.config.k_critical} | "
                      f"Ent: {avg_entropy:.2f} | T: {temp:.2f} {status}")
            
            # Evaluation
            if (i + 1) % self.config.eval_interval == 0:
                eval_results = self.evaluate()
                self._print_eval(eval_results, i + 1)
                
                # Save best model
                if eval_results['cfr_rl'] < self.best_performance:
                    self.best_performance = eval_results['cfr_rl']
                    self.save("best_abilene_model.pt")
        
        elapsed = time.time() - start_time
        print(f"\nTraining complete in {elapsed/60:.1f} minutes")
        
        return self.history
    
    def evaluate(self, n_episodes: int = None) -> Dict[str, float]:
        """Evaluate all methods."""
        n_episodes = n_episodes or self.config.eval_episodes
        
        results = {
            'single_path': [],
            'random_k': [],
            'top_k': [],
            'cfr_rl': [],
            'ecmp': []
        }
        
        for _ in range(n_episodes):
            # Use diverse traffic for evaluation
            pattern = np.random.choice(['bimodal', 'gravity', 'hotspot'])
            if pattern == 'bimodal':
                tm = self.traffic_gen.generate_bimodal()
            elif pattern == 'gravity':
                tm = self.traffic_gen.generate_gravity()
            else:
                tm = self.traffic_gen.generate_hotspot()
            
            features = self.prepare_features(tm)
            
            results['single_path'].append(self.solver.solve_single_path(tm))
            results['ecmp'].append(self.solver.solve_ecmp(tm))
            
            util_random, _ = self.solver.solve_random_k(tm, self.config.k_critical)
            results['random_k'].append(util_random)
            
            util_top_k, _ = self.solver.solve_top_k(tm, self.config.k_critical)
            results['top_k'].append(util_top_k)
            
            # Policy selection (deterministic for eval)
            selected, _, _ = self.policy.select_flows(
                features, self.config.k_critical,
                deterministic=True,
                mask=self.is_multi_path
            )
            util_cfr, _ = self.solver.solve(tm, selected, background_routing="single_path")
            results['cfr_rl'].append(util_cfr)
        
        return {k: np.mean(v) for k, v in results.items()}
    
    def _print_eval(self, results: Dict[str, float], iteration: int):
        """Print evaluation results."""
        print(f"\n  --- Eval @ iter {iteration} ---")
        
        single = results['single_path']
        ecmp = results['ecmp']
        cfr = results['cfr_rl']
        topk = results['top_k']
        
        for method in ['single_path', 'random_k', 'top_k', 'cfr_rl', 'ecmp']:
            util = results[method]
            vs_single = (single - util) / single * 100
            print(f"    {method:<12}: {util:.4f} ({vs_single:+.1f}% vs single)")
        
        if cfr < topk:
            print(f"  ✓ CFR-RL beats Top-K by {(topk-cfr)/topk*100:.1f}%")
        elif cfr > topk * 1.01:
            print(f"  ✗ Top-K beats CFR-RL by {(cfr-topk)/topk*100:.1f}%")
        else:
            print(f"  = CFR-RL matches Top-K")
        print()
    
    def save(self, path: str):
        """Save model."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'config': self.config,
            'history': self.history[-1000:],  # Save recent history
        }, path)
    
    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        print(f"Loaded model from {path}")


def main():
    """Main training script."""
    print("="*60)
    print("CFR-RL ON ABILENE - PURE REINFORCEMENT LEARNING")
    print("="*60)
    
    # Configuration
    config = TrainingConfig(
        k_critical=8,
        hidden_dim=64,
        total_iterations=10000,
        lr=3e-4,
        entropy_coef=0.05,
        temperature=1.0,
        temperature_decay=0.9995,
        min_temperature=0.1,
        eval_interval=1000,
        log_interval=200,
    )
    
    trainer = AbileneRLTrainer(config, seed=42)
    
    # Show topology
    trainer.topo.print_summary()
    
    # Initial evaluation
    print("\n" + "="*60)
    print("BEFORE TRAINING")
    print("="*60)
    results_before = trainer.evaluate()
    trainer._print_eval(results_before, 0)
    
    # Train
    trainer.train()
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    results_after = trainer.evaluate(n_episodes=100)
    trainer._print_eval(results_after, config.total_iterations)
    
    # Save final model
    trainer.save("final_abilene_model.pt")
    
    return trainer, results_after


if __name__ == "__main__":
    trainer, results = main()