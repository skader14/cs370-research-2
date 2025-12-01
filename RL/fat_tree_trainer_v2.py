"""
CFR-RL Trainer for Fat-Tree Topology - V2

Key change: Pointwise scoring instead of distribution learning.
- Learn a score for each flow (should this flow be critical?)
- Select top-K flows by score
- Much easier to learn than full distribution

This is essentially learning to rank/classify flows.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from fat_tree_topology import FatTreeTopology
from fat_tree_lp_solver import FatTreeLPSolver
from fat_tree_traffic_gen import FatTreeTrafficGenerator


class PointwisePolicy(nn.Module):
    """
    Pointwise scoring network.
    
    For each flow, outputs a score indicating how "critical" it is.
    Selection: take top-K flows by score.
    
    This is much easier to learn than a full distribution.
    """
    
    def __init__(self, num_flows: int, hidden_dim: int = 64):
        super().__init__()
        self.num_flows = num_flows
        
        # Per-flow feature processing
        # Input per flow: [demand, num_paths, is_cross_pod]
        self.flow_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Score head: outputs scalar score per flow
        self.score_head = nn.Linear(hidden_dim, 1)
        
        # Value head for RL baseline
        self.value_head = nn.Sequential(
            nn.Linear(num_flows * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, flow_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            flow_features: [batch, num_flows, 3] - features per flow
            
        Returns:
            scores: [batch, num_flows] - score per flow
            value: [batch, 1] - state value
        """
        batch_size = flow_features.shape[0]
        
        # Encode each flow
        # [batch, num_flows, 3] -> [batch, num_flows, hidden]
        encoded = self.flow_encoder(flow_features)
        
        # Score each flow
        # [batch, num_flows, hidden] -> [batch, num_flows, 1] -> [batch, num_flows]
        scores = self.score_head(encoded).squeeze(-1)
        
        # Value from all flow encodings
        flat_encoded = encoded.view(batch_size, -1)
        value = self.value_head(flat_encoded)
        
        return scores, value
    
    def select_flows(self, flow_features: np.ndarray, k: int,
                    deterministic: bool = False,
                    temperature: float = 1.0) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
        """
        Select k critical flows.
        
        Args:
            flow_features: [num_flows, 3]
            k: Number of flows to select
            deterministic: If True, take top-k by score
            temperature: Sampling temperature (lower = more greedy)
            
        Returns:
            selected: List of selected flow indices
            log_prob: Log probability of selection
            entropy: Entropy of selection distribution
        """
        x = torch.FloatTensor(flow_features).unsqueeze(0)  # [1, num_flows, 3]
        scores, value = self.forward(x)
        scores = scores.squeeze(0)  # [num_flows]
        
        if deterministic:
            # Take top-k by score
            selected = torch.topk(scores, k).indices.tolist()
            log_prob = torch.tensor(0.0)  # Deterministic, no log prob
            entropy = torch.tensor(0.0)
        else:
            # Sample using softmax over scores
            probs = torch.softmax(scores / temperature, dim=0)
            selected = torch.multinomial(probs, k, replacement=False).tolist()
            
            # Log probability
            log_probs = torch.log_softmax(scores / temperature, dim=0)
            log_prob = log_probs[selected].sum()
            
            # Entropy
            entropy = -(probs * log_probs).sum()
        
        return selected, log_prob, entropy


@dataclass  
class TrainerConfig:
    """Training configuration."""
    k_critical: int = 8
    hidden_dim: int = 64
    lr: float = 1e-3
    pretrain_lr: float = 1e-2
    entropy_coef: float = 0.01
    grad_clip: float = 1.0
    pretrain_iters: int = 1000
    rl_iters: int = 500
    temperature: float = 0.5  # Sampling temperature


class FatTreeTrainerV2:
    """Improved trainer with pointwise scoring."""
    
    def __init__(self, config: TrainerConfig = None, seed: int = 42):
        self.config = config or TrainerConfig()
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Initialize components
        self.topo = FatTreeTopology(k=4)
        self.solver = FatTreeLPSolver(self.topo)
        self.traffic_gen = FatTreeTrafficGenerator(self.topo, seed=seed)
        
        # Policy network
        self.policy = PointwisePolicy(
            num_flows=self.topo.num_flows,
            hidden_dim=self.config.hidden_dim
        )
        
        # Precompute flow metadata
        self._compute_flow_metadata()
        
    def _compute_flow_metadata(self):
        """Compute static metadata for each flow."""
        self.num_paths = np.zeros(self.topo.num_flows)
        self.is_cross_pod = np.zeros(self.topo.num_flows)
        self.is_multi_path = np.zeros(self.topo.num_flows)
        
        for flow_idx, (src, dst) in enumerate(self.topo.flow_pairs):
            self.num_paths[flow_idx] = self.topo.get_num_paths(src, dst)
            self.is_cross_pod[flow_idx] = float(not self.topo.hosts_same_pod(src, dst))
            self.is_multi_path[flow_idx] = float(self.num_paths[flow_idx] > 1)
    
    def prepare_features(self, tm: np.ndarray) -> np.ndarray:
        """
        Prepare per-flow features from traffic matrix.
        
        Returns: [num_flows, 3] array with [demand, num_paths, is_cross_pod] per flow
        """
        flows = self.traffic_gen.tm_to_flow_vector(tm)
        
        # Normalize demands
        max_flow = flows.max() if flows.max() > 0 else 1.0
        normalized_demands = flows / max_flow
        
        # Stack features: [num_flows, 3]
        features = np.stack([
            normalized_demands,
            self.num_paths / 4.0,  # Normalize path count
            self.is_cross_pod
        ], axis=1)
        
        return features
    
    def get_top_k_indices(self, tm: np.ndarray) -> List[int]:
        """Get indices of top-K multi-path flows by demand."""
        flows = self.traffic_gen.tm_to_flow_vector(tm)
        masked_flows = flows * self.is_multi_path
        top_k = np.argsort(-masked_flows)[:self.config.k_critical]
        return top_k.tolist()
    
    def supervised_pretrain(self, n_iters: int = None, log_interval: int = 100) -> List[float]:
        """
        Pre-train with pointwise binary cross-entropy.
        
        Target: 1 for flows in Top-K, 0 for others.
        Loss: BCE on sigmoid(scores)
        """
        n_iters = n_iters or self.config.pretrain_iters
        print(f"Supervised pre-training ({n_iters} iterations)...")
        
        optimizer = optim.Adam(self.policy.parameters(), lr=self.config.pretrain_lr)
        criterion = nn.BCEWithLogitsLoss()
        
        losses = []
        
        for i in range(n_iters):
            # Generate traffic
            tm = self.traffic_gen.generate_bimodal()
            features = self.prepare_features(tm)
            
            # Get top-K target
            top_k = set(self.get_top_k_indices(tm))
            
            # Create binary target
            target = torch.zeros(self.topo.num_flows)
            for idx in top_k:
                target[idx] = 1.0
            
            # Forward pass
            x = torch.FloatTensor(features).unsqueeze(0)  # [1, num_flows, 3]
            scores, _ = self.policy(x)
            scores = scores.squeeze(0)  # [num_flows]
            
            # BCE loss
            loss = criterion(scores, target)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.grad_clip)
            optimizer.step()
            
            losses.append(loss.item())
            
            if (i + 1) % log_interval == 0:
                avg_loss = np.mean(losses[-log_interval:])
                
                # Check overlap with Top-K
                with torch.no_grad():
                    selected, _, _ = self.policy.select_flows(features, self.config.k_critical, True)
                    overlap = len(set(selected) & top_k)
                
                # Also check how well scores correlate with demands
                flows = self.traffic_gen.tm_to_flow_vector(tm)
                with torch.no_grad():
                    x = torch.FloatTensor(features).unsqueeze(0)
                    scores_np = self.policy(x)[0].squeeze(0).numpy()
                
                # Correlation between scores and (demand * is_multi_path)
                importance = flows * self.is_multi_path
                corr = np.corrcoef(scores_np, importance)[0, 1] if importance.std() > 0 else 0
                
                print(f"  Iter {i+1}/{n_iters} | Loss: {avg_loss:.4f} | Overlap: {overlap}/{self.config.k_critical} | Corr: {corr:.3f}")
        
        return losses
    
    def train_step(self) -> Dict[str, float]:
        """Single RL training step."""
        tm = self.traffic_gen.generate_bimodal()
        features = self.prepare_features(tm)
        
        # Baselines
        util_single = self.solver.solve_single_path(tm)
        util_top_k, _ = self.solver.solve_top_k(tm, self.config.k_critical)
        
        # Policy selection
        selected, log_prob, entropy = self.policy.select_flows(
            features, self.config.k_critical, 
            deterministic=False,
            temperature=self.config.temperature
        )
        
        # Evaluate
        util_policy, _ = self.solver.solve(tm, selected, background_routing="single_path")
        
        # Reward: improvement over single-path, normalized
        reward = (util_single - util_policy) / util_single
        
        # Baseline: Top-K performance
        reward_top_k = (util_single - util_top_k) / util_single
        advantage = reward - reward_top_k
        
        # Loss
        policy_loss = -log_prob * advantage
        entropy_loss = -self.config.entropy_coef * entropy
        loss = policy_loss + entropy_loss
        
        return {
            'loss': loss,
            'util_policy': util_policy,
            'util_top_k': util_top_k,
            'util_single': util_single,
            'advantage': advantage,
            'entropy': entropy.item()
        }
    
    def train_rl(self, n_iters: int = None, log_interval: int = 100) -> List[Dict]:
        """RL fine-tuning."""
        n_iters = n_iters or self.config.rl_iters
        print(f"RL fine-tuning ({n_iters} iterations)...")
        
        optimizer = optim.Adam(self.policy.parameters(), lr=self.config.lr)
        history = []
        
        for i in range(n_iters):
            result = self.train_step()
            
            optimizer.zero_grad()
            result['loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.grad_clip)
            optimizer.step()
            
            history.append({
                'util_policy': result['util_policy'],
                'util_top_k': result['util_top_k'],
                'entropy': result['entropy']
            })
            
            if (i + 1) % log_interval == 0:
                recent = history[-log_interval:]
                avg_policy = np.mean([h['util_policy'] for h in recent])
                avg_top_k = np.mean([h['util_top_k'] for h in recent])
                avg_entropy = np.mean([h['entropy'] for h in recent])
                
                # Win rate
                wins = sum(1 for h in recent if h['util_policy'] <= h['util_top_k'])
                win_rate = wins / len(recent) * 100
                
                status = "✓" if avg_policy <= avg_top_k else "✗"
                print(f"  Iter {i+1}/{n_iters} | CFR-RL: {avg_policy:.3f} | Top-K: {avg_top_k:.3f} | Win%: {win_rate:.0f}% {status}")
        
        return history
    
    def train(self, do_pretrain: bool = True) -> Dict:
        """Full training pipeline."""
        history = {'pretrain': [], 'rl': []}
        
        if do_pretrain:
            history['pretrain'] = self.supervised_pretrain()
        
        history['rl'] = self.train_rl()
        
        return history
    
    def evaluate(self, n_episodes: int = 50) -> Dict[str, float]:
        """Evaluate all methods."""
        results = {
            'single_path': [],
            'random_k': [],
            'top_k': [],
            'cfr_rl': [],
            'ecmp': []
        }
        
        for _ in range(n_episodes):
            tm = self.traffic_gen.generate_bimodal()
            features = self.prepare_features(tm)
            
            results['single_path'].append(self.solver.solve_single_path(tm))
            results['ecmp'].append(self.solver.solve_ecmp(tm))
            
            util_random, _ = self.solver.solve_random_k(tm, self.config.k_critical)
            results['random_k'].append(util_random)
            
            util_top_k, _ = self.solver.solve_top_k(tm, self.config.k_critical)
            results['top_k'].append(util_top_k)
            
            selected, _, _ = self.policy.select_flows(features, self.config.k_critical, deterministic=True)
            util_cfr, _ = self.solver.solve(tm, selected, background_routing="single_path")
            results['cfr_rl'].append(util_cfr)
        
        return {k: np.mean(v) for k, v in results.items()}
    
    def print_evaluation(self, results: Dict[str, float]):
        """Pretty print results."""
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        single = results['single_path']
        ecmp = results['ecmp']
        
        print(f"\n{'Method':<15} {'Util':>8} {'vs Single':>12} {'ECMP Gap':>10}")
        print("-"*50)
        
        for method in ['single_path', 'random_k', 'top_k', 'cfr_rl', 'ecmp']:
            util = results[method]
            vs_single = (single - util) / single * 100
            
            if method in ['single_path', 'ecmp']:
                ecmp_gap = "-"
            else:
                max_improvement = single - ecmp
                actual_improvement = single - util
                ecmp_gap = f"{actual_improvement/max_improvement*100:.0f}%"
            
            print(f"{method:<15} {util:>8.4f} {vs_single:>+11.1f}% {ecmp_gap:>10}")
        
        cfr_rl = results['cfr_rl']
        top_k = results['top_k']
        
        print("\n" + "-"*50)
        if cfr_rl <= top_k:
            pct_better = (top_k - cfr_rl) / top_k * 100
            print(f"✓ CFR-RL ({cfr_rl:.4f}) beats Top-K ({top_k:.4f}) by {pct_better:.1f}%")
        else:
            pct_worse = (cfr_rl - top_k) / top_k * 100
            print(f"✗ Top-K ({top_k:.4f}) beats CFR-RL ({cfr_rl:.4f}) by {pct_worse:.1f}%")
    
    def save(self, path: str):
        """Save model."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'config': self.config
        }, path)
        print(f"Saved to {path}")
    
    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        print(f"Loaded from {path}")


def main():
    print("="*60)
    print("CFR-RL V2 - POINTWISE SCORING")
    print("="*60)
    
    config = TrainerConfig(
        k_critical=8,
        hidden_dim=64,
        lr=1e-3,
        pretrain_lr=1e-2,
        entropy_coef=0.01,
        pretrain_iters=1000,
        rl_iters=500,
        temperature=0.5
    )
    
    trainer = FatTreeTrainerV2(config, seed=42)
    trainer.topo.print_summary()
    
    # Before
    print("\n" + "="*60)
    print("BEFORE TRAINING")
    print("="*60)
    results_before = trainer.evaluate()
    trainer.print_evaluation(results_before)
    
    # Train
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    trainer.train(do_pretrain=True)
    
    # After
    print("\n" + "="*60)
    print("AFTER TRAINING")
    print("="*60)
    results_after = trainer.evaluate()
    trainer.print_evaluation(results_after)
    
    trainer.save("cfr_rl_v2.pt")
    
    return trainer, results_after


if __name__ == "__main__":
    trainer, results = main()