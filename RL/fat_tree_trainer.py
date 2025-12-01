"""
CFR-RL Trainer for Fat-Tree Topology

Training approach:
1. Supervised pre-training: Learn to imitate Top-K heuristic
2. RL fine-tuning: Improve beyond Top-K using REINFORCE

Key insight: Fat-tree has 4 paths for cross-pod flows, giving RL
more optimization opportunity than simple dual-core topology.
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


class PolicyNetwork(nn.Module):
    """
    Policy network for selecting critical flows.
    
    Input: Flow demands + flow metadata (path count, flow type)
    Output: Probability distribution over flows
    """
    
    def __init__(self, num_flows: int, hidden_dim: int = 128):
        super().__init__()
        # Input: flow_demand + num_paths + is_cross_pod for each flow
        input_dim = num_flows * 3
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_flows)
        )
        
        # Value head for baseline
        self.value_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input state [batch, input_dim]
            
        Returns:
            log_probs: Log probabilities [batch, num_flows]
            value: State value [batch, 1]
        """
        logits = self.net(x)
        log_probs = torch.log_softmax(logits, dim=-1)
        value = self.value_head(x)
        return log_probs, value
    
    def select_flows(self, state: np.ndarray, k: int, 
                    deterministic: bool = False) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
        """
        Select k critical flows.
        
        Args:
            state: Input state
            k: Number of flows to select
            deterministic: If True, select top-k by probability
            
        Returns:
            selected: List of selected flow indices
            log_prob: Log probability of selection
            entropy: Entropy of distribution
        """
        x = torch.FloatTensor(state).unsqueeze(0)
        log_probs, value = self.forward(x)
        probs = torch.exp(log_probs.squeeze(0))
        
        if deterministic:
            # Select top-k by probability
            selected = torch.topk(probs, k).indices.tolist()
        else:
            # Sample without replacement
            selected = torch.multinomial(probs, k, replacement=False).tolist()
        
        # Log probability of selection (sum of log probs)
        log_prob = log_probs.squeeze(0)[selected].sum()
        
        # Entropy
        entropy = -(probs * log_probs.squeeze(0)).sum()
        
        return selected, log_prob, entropy


@dataclass
class TrainerConfig:
    """Training configuration."""
    k_critical: int = 8          # Number of critical flows
    hidden_dim: int = 128        # Policy hidden dimension
    lr: float = 1e-3             # Learning rate
    pretrain_lr: float = 5e-3    # Pre-training learning rate
    entropy_coef: float = 0.02   # Entropy bonus coefficient
    grad_clip: float = 0.5       # Gradient clipping
    pretrain_iters: int = 500    # Pre-training iterations
    rl_iters: int = 500          # RL iterations
    

class FatTreeTrainer:
    """Trainer for CFR-RL on fat-tree topology."""
    
    def __init__(self, config: TrainerConfig = None, seed: int = 42):
        self.config = config or TrainerConfig()
        self.seed = seed
        
        # Initialize components
        self.topo = FatTreeTopology(k=4)
        self.solver = FatTreeLPSolver(self.topo)
        self.traffic_gen = FatTreeTrafficGenerator(self.topo, seed=seed)
        
        # Policy network
        self.policy = PolicyNetwork(
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
            
    def prepare_state(self, tm: np.ndarray) -> np.ndarray:
        """
        Prepare policy input state from traffic matrix.
        
        State = [normalized_demands, num_paths, is_cross_pod]
        """
        flows = self.traffic_gen.tm_to_flow_vector(tm)
        
        # Normalize demands
        max_flow = flows.max() if flows.max() > 0 else 1.0
        normalized = flows / max_flow
        
        # Normalize path counts
        normalized_paths = self.num_paths / 4.0  # Max is 4 paths
        
        # Concatenate features
        state = np.concatenate([normalized, normalized_paths, self.is_cross_pod])
        return state
    
    def get_top_k_indices(self, tm: np.ndarray) -> List[int]:
        """Get indices of top-K multi-path flows by demand."""
        flows = self.traffic_gen.tm_to_flow_vector(tm)
        
        # Only consider multi-path flows
        masked_flows = flows * self.is_multi_path
        
        # Get top-K
        top_k = np.argsort(-masked_flows)[:self.config.k_critical]
        return top_k.tolist()
    
    def supervised_pretrain(self, n_iters: int = None, log_interval: int = 100) -> List[float]:
        """
        Pre-train policy to imitate Top-K selection.
        
        Uses soft targets proportional to (demand * is_multi_path).
        """
        n_iters = n_iters or self.config.pretrain_iters
        print(f"Supervised pre-training ({n_iters} iterations)...")
        
        optimizer = optim.Adam(self.policy.parameters(), lr=self.config.pretrain_lr)
        losses = []
        
        for i in range(n_iters):
            # Generate traffic
            tm = self.traffic_gen.generate_bimodal()
            state = self.prepare_state(tm)
            flows = self.traffic_gen.tm_to_flow_vector(tm)
            
            # Create soft target: importance = demand * multi_path_mask
            importance = flows * self.is_multi_path
            
            # Temperature-scaled softmax (lower temp = more peaked)
            temperature = 0.1
            if importance.max() > 0:
                importance_scaled = importance / importance.max()
            else:
                importance_scaled = importance
            target_logits = importance_scaled / temperature
            target_probs = torch.softmax(torch.FloatTensor(target_logits), dim=0)
            
            # Forward pass
            x = torch.FloatTensor(state).unsqueeze(0)
            log_probs, _ = self.policy(x)
            probs = torch.exp(log_probs.squeeze(0))
            
            # KL divergence loss
            loss = torch.sum(target_probs * (torch.log(target_probs + 1e-10) - log_probs.squeeze(0)))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.grad_clip)
            optimizer.step()
            
            losses.append(loss.item())
            
            if (i + 1) % log_interval == 0:
                avg_loss = np.mean(losses[-log_interval:])
                
                # Check overlap with Top-K
                with torch.no_grad():
                    selected, _, _ = self.policy.select_flows(state, self.config.k_critical, True)
                    top_k = self.get_top_k_indices(tm)
                    overlap = len(set(selected) & set(top_k))
                
                print(f"  Iter {i+1}/{n_iters} | Loss: {avg_loss:.4f} | Top-K overlap: {overlap}/{self.config.k_critical}")
        
        return losses
    
    def train_step(self, tm: np.ndarray = None) -> Dict[str, float]:
        """
        Single RL training step.
        
        Returns dict with metrics.
        """
        # Generate traffic if not provided
        if tm is None:
            tm = self.traffic_gen.generate_bimodal()
            
        state = self.prepare_state(tm)
        
        # Get baseline utilizations
        util_single = self.solver.solve_single_path(tm)
        util_top_k, top_k_flows = self.solver.solve_top_k(tm, self.config.k_critical)
        
        # Policy selects flows
        selected, log_prob, entropy = self.policy.select_flows(
            state, self.config.k_critical, deterministic=False
        )
        
        # Evaluate policy selection
        util_policy, _ = self.solver.solve(tm, selected, background_routing="single_path")
        
        # Reward: relative improvement over single-path
        # Higher is better (lower utilization = higher reward)
        reward = (util_single - util_policy) / util_single
        
        # Advantage: beat 90% of Top-K performance
        reward_top_k = (util_single - util_top_k) / util_single
        advantage = reward - 0.9 * reward_top_k
        
        # Policy gradient loss
        policy_loss = -log_prob * advantage
        entropy_loss = -self.config.entropy_coef * entropy
        loss = policy_loss + entropy_loss
        
        return {
            'loss': loss,
            'util_policy': util_policy,
            'util_top_k': util_top_k,
            'util_single': util_single,
            'reward': reward,
            'advantage': advantage,
            'entropy': entropy.item(),
            'selected': selected
        }
    
    def train_rl(self, n_iters: int = None, log_interval: int = 100) -> List[Dict]:
        """
        RL fine-tuning phase.
        """
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
                
                status = "✓" if avg_policy <= avg_top_k else "✗"
                print(f"  Iter {i+1}/{n_iters} | CFR-RL: {avg_policy:.3f} | Top-K: {avg_top_k:.3f} | Entropy: {avg_entropy:.2f} {status}")
        
        return history
    
    def train(self, do_pretrain: bool = True) -> Dict:
        """
        Full training pipeline.
        
        Returns training history.
        """
        history = {'pretrain': [], 'rl': []}
        
        if do_pretrain:
            history['pretrain'] = self.supervised_pretrain()
            
        history['rl'] = self.train_rl()
        
        return history
    
    def evaluate(self, n_episodes: int = 20) -> Dict[str, float]:
        """
        Evaluate all methods.
        
        Returns average utilization for each method.
        """
        results = {
            'single_path': [],
            'random_k': [],
            'top_k': [],
            'cfr_rl': [],
            'ecmp': []
        }
        
        for _ in range(n_episodes):
            tm = self.traffic_gen.generate_bimodal()
            state = self.prepare_state(tm)
            
            # Single path
            results['single_path'].append(self.solver.solve_single_path(tm))
            
            # ECMP
            results['ecmp'].append(self.solver.solve_ecmp(tm))
            
            # Random-K
            util_random, _ = self.solver.solve_random_k(tm, self.config.k_critical)
            results['random_k'].append(util_random)
            
            # Top-K
            util_top_k, _ = self.solver.solve_top_k(tm, self.config.k_critical)
            results['top_k'].append(util_top_k)
            
            # CFR-RL
            selected, _, _ = self.policy.select_flows(state, self.config.k_critical, deterministic=True)
            util_cfr, _ = self.solver.solve(tm, selected, background_routing="single_path")
            results['cfr_rl'].append(util_cfr)
        
        return {k: np.mean(v) for k, v in results.items()}
    
    def print_evaluation(self, results: Dict[str, float]):
        """Pretty print evaluation results."""
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
            
            if method == 'single_path':
                ecmp_gap = "-"
            elif method == 'ecmp':
                ecmp_gap = "-"
            else:
                # How much of the improvement does this method capture?
                max_improvement = single - ecmp
                actual_improvement = single - util
                ecmp_gap = f"{actual_improvement/max_improvement*100:.0f}%"
            
            print(f"{method:<15} {util:>8.4f} {vs_single:>+11.1f}% {ecmp_gap:>10}")
        
        # Summary
        cfr_rl = results['cfr_rl']
        top_k = results['top_k']
        
        print("\n" + "-"*50)
        if cfr_rl <= top_k:
            print(f"✓ CFR-RL ({cfr_rl:.4f}) beats or matches Top-K ({top_k:.4f})")
        else:
            print(f"✗ Top-K ({top_k:.4f}) still better than CFR-RL ({cfr_rl:.4f})")
    
    def save(self, path: str):
        """Save trained policy."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'config': self.config,
            'flow_metadata': {
                'num_paths': self.num_paths,
                'is_cross_pod': self.is_cross_pod,
                'is_multi_path': self.is_multi_path
            }
        }, path)
        print(f"Saved model to {path}")
    
    def load(self, path: str):
        """Load trained policy."""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        print(f"Loaded model from {path}")


def main():
    """Main training script."""
    print("="*60)
    print("CFR-RL TRAINING ON FAT-TREE TOPOLOGY")
    print("="*60)
    
    # Configuration
    config = TrainerConfig(
        k_critical=8,
        hidden_dim=128,
        lr=1e-3,
        pretrain_lr=5e-3,
        entropy_coef=0.02,
        pretrain_iters=500,
        rl_iters=500
    )
    
    trainer = FatTreeTrainer(config, seed=42)
    
    # Show topology info
    trainer.topo.print_summary()
    
    # Evaluate before training
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
    
    # Evaluate after training
    print("\n" + "="*60)
    print("AFTER TRAINING")
    print("="*60)
    results_after = trainer.evaluate()
    trainer.print_evaluation(results_after)
    
    # Save model
    trainer.save("cfr_rl_fat_tree.pt")
    
    return trainer, results_after


if __name__ == "__main__":
    trainer, results = main()