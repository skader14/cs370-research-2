"""
REINFORCE trainer with supervised pre-training for Critical Flow Selection.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import deque
import os

from topology import Topology
from traffic_gen import TrafficGenerator
from lp_solver import LPSolver
from policy import PolicyWithBaseline


class REINFORCETrainer:
    """
    Trainer that combines supervised pre-training with REINFORCE fine-tuning.
    """
    
    def __init__(self, 
                 topology: Topology,
                 k_critical: int = 8,
                 hidden_dim: int = 128,
                 lr: float = 1e-3,
                 entropy_coef: float = 0.05,
                 device: str = "cpu"):
        self.topo = topology
        self.k = k_critical
        self.device = device
        self.entropy_coef = entropy_coef
        
        self.n_flows = topology.get_n_flows()
        self.traffic_gen = TrafficGenerator(topology)
        self.lp_solver = LPSolver(topology)
        
        # Policy network
        self.policy = PolicyWithBaseline(self.n_flows, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Precompute which flows are cross-edge (have 2 paths)
        self.cross_edge_mask = self._compute_cross_edge_mask()
        
    def _compute_cross_edge_mask(self) -> np.ndarray:
        """Create mask indicating which flows are cross-edge (optimizable)."""
        mask = np.zeros(self.n_flows)
        host_ids = self.topo.host_ids
        
        idx = 0
        for i in range(self.topo.n_hosts):
            for j in range(self.topo.n_hosts):
                if i != j:
                    src_host = host_ids[i]
                    dst_host = host_ids[j]
                    if not self.topo.hosts_same_edge(src_host, dst_host):
                        mask[idx] = 1.0
                    idx += 1
        return mask
        
    def flow_idx_to_pair(self, idx: int) -> Tuple[int, int]:
        """Convert flat flow index to (src, dst) pair."""
        n = self.topo.n_hosts
        src = idx // (n - 1)
        dst_offset = idx % (n - 1)
        dst = dst_offset if dst_offset < src else dst_offset + 1
        return (src, dst)
    
    def prepare_state(self, tm: np.ndarray) -> np.ndarray:
        """Prepare state for policy network."""
        flows = self.traffic_gen.tm_to_flow_vector(tm)
        total = flows.sum()
        if total > 0:
            normalized = flows / total
        else:
            normalized = flows
        return normalized
    
    def get_top_k_indices(self, tm: np.ndarray) -> List[int]:
        """Get indices of top-K flows by demand (cross-edge only)."""
        flows = self.traffic_gen.tm_to_flow_vector(tm)
        # Mask out same-edge flows (they can't be optimized)
        masked_flows = flows * self.cross_edge_mask
        # Get top-K indices
        top_k = np.argsort(masked_flows)[-self.k:][::-1].tolist()
        return top_k
    
    def supervised_pretrain(self, n_iterations: int = 500, 
                           log_interval: int = 100,
                           **traffic_kwargs) -> List[float]:
        """
        Pre-train policy to imitate Top-K selection.
        This gives the policy a good starting point before RL fine-tuning.
        """
        print("Supervised pre-training to imitate Top-K...")
        losses = []
        
        for i in range(n_iterations):
            # Generate traffic
            tm = self.traffic_gen.generate_bimodal(**traffic_kwargs)
            state = self.prepare_state(tm)
            
            # Get target: top-K indices
            target_indices = self.get_top_k_indices(tm)
            
            # Create target distribution: high probability on top-K flows
            target_probs = torch.zeros(self.n_flows)
            for idx in target_indices:
                target_probs[idx] = 1.0 / self.k
            
            # Forward pass
            x = torch.FloatTensor(state).unsqueeze(0)
            log_probs, _ = self.policy(x)
            probs = torch.exp(log_probs.squeeze(0))
            
            # Cross-entropy loss
            loss = -torch.sum(target_probs * torch.log(probs + 1e-10))
            
            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            losses.append(loss.item())
            
            if (i + 1) % log_interval == 0:
                avg_loss = np.mean(losses[-log_interval:])
                print(f"  Pretrain {i+1}/{n_iterations} | Loss: {avg_loss:.4f}")
        
        return losses
    
    def train_step(self, traffic_matrix: np.ndarray) -> Dict:
        """Single RL training step."""
        state = self.prepare_state(traffic_matrix)
        
        # Baseline: single-path (no optimization)
        util_baseline, _ = self.lp_solver.solve(traffic_matrix, [])
        
        # Select flows using policy
        selected_indices, log_prob, value = self.policy.select_critical_flows(
            state, self.k, deterministic=False
        )
        critical_flows = [self.flow_idx_to_pair(idx) for idx in selected_indices]
        
        # Compute utilization with selected flows
        util_optimized, _ = self.lp_solver.solve(traffic_matrix, critical_flows)
        
        # Reward: reduction in utilization (higher is better)
        # Scale to make gradients reasonable
        reward = (util_baseline - util_optimized) / util_baseline
        
        # Compare to Top-K for reference
        top_k_indices = self.get_top_k_indices(traffic_matrix)
        top_k_flows = [self.flow_idx_to_pair(idx) for idx in top_k_indices]
        util_top_k, _ = self.lp_solver.solve(traffic_matrix, top_k_flows)
        reward_top_k = (util_baseline - util_top_k) / util_baseline
        
        # Advantage: how much better/worse than Top-K?
        advantage = reward - reward_top_k * 0.9  # Target: beat 90% of Top-K performance
        
        # Policy loss
        policy_loss = -log_prob * advantage.detach() if isinstance(advantage, torch.Tensor) else -log_prob * advantage
        
        # Value loss
        value_target = torch.tensor(reward)
        value_loss = 0.5 * (value_target - value) ** 2
        
        # Entropy bonus
        probs = self.policy.get_probs(torch.FloatTensor(state))
        entropy = -(probs * torch.log(probs + 1e-10)).sum()
        
        # Total loss
        loss = policy_loss + value_loss - self.entropy_coef * entropy
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()
        
        return {
            "loss": loss.item(),
            "reward": reward,
            "util_rl": util_optimized,
            "util_top_k": util_top_k,
            "util_baseline": util_baseline,
            "entropy": entropy.item(),
        }
    
    def train(self, n_iterations: int, 
             traffic_method: str = "bimodal",
             log_interval: int = 100,
             save_path: Optional[str] = None,
             pretrain: bool = True,
             pretrain_iterations: int = 500,
             **traffic_kwargs) -> List[Dict]:
        """
        Main training loop with optional pre-training.
        """
        # Supervised pre-training
        if pretrain:
            self.supervised_pretrain(pretrain_iterations, 100, **traffic_kwargs)
            print()
        
        # RL fine-tuning
        print("RL fine-tuning...")
        stats = []
        
        for i in range(n_iterations):
            if traffic_method == "bimodal":
                tm = self.traffic_gen.generate_bimodal(**traffic_kwargs)
            else:
                tm = self.traffic_gen.generate_exponential(**traffic_kwargs)
            
            step_stats = self.train_step(tm)
            step_stats["iteration"] = i
            stats.append(step_stats)
            
            if (i + 1) % log_interval == 0:
                recent = stats[-log_interval:]
                avg_util_rl = np.mean([s["util_rl"] for s in recent])
                avg_util_topk = np.mean([s["util_top_k"] for s in recent])
                entropy = step_stats["entropy"]
                
                print(f"  Iter {i+1}/{n_iterations} | "
                      f"CFR-RL: {avg_util_rl:.3f} | "
                      f"Top-K: {avg_util_topk:.3f} | "
                      f"Entropy: {entropy:.2f}")
        
        if save_path:
            self.save(save_path)
            
        return stats
    
    def evaluate(self, n_episodes: int = 50, 
                traffic_method: str = "bimodal",
                **traffic_kwargs) -> Dict:
        """Evaluate trained policy against baselines."""
        results = {
            "single_path": [],
            "random_k": [],
            "top_k": [],
            "cfr_rl": [],
            "ecmp": [],
        }
        
        for _ in range(n_episodes):
            if traffic_method == "bimodal":
                tm = self.traffic_gen.generate_bimodal(**traffic_kwargs)
            else:
                tm = self.traffic_gen.generate_exponential(**traffic_kwargs)
            
            state = self.prepare_state(tm)
            flows_flat = self.traffic_gen.tm_to_flow_vector(tm)
            
            # Single-Path
            util_single, _ = self.lp_solver.solve(tm, [])
            results["single_path"].append(util_single)
            
            # ECMP
            util_ecmp, _ = self.lp_solver.solve_ecmp(tm)
            results["ecmp"].append(util_ecmp)
            
            # CFR-RL
            selected_indices, _, _ = self.policy.select_critical_flows(
                state, self.k, deterministic=True
            )
            critical_flows = [self.flow_idx_to_pair(idx) for idx in selected_indices]
            util_cfr, _ = self.lp_solver.solve(tm, critical_flows)
            results["cfr_rl"].append(util_cfr)
            
            # Top-K (cross-edge only)
            top_k_indices = self.get_top_k_indices(tm)
            top_k_flows = [self.flow_idx_to_pair(idx) for idx in top_k_indices]
            util_top_k, _ = self.lp_solver.solve(tm, top_k_flows)
            results["top_k"].append(util_top_k)
            
            # Random-K (from non-zero flows)
            nonzero = np.where(flows_flat > 0)[0]
            if len(nonzero) >= self.k:
                random_idx = np.random.choice(nonzero, self.k, replace=False)
            else:
                random_idx = nonzero
            random_flows = [self.flow_idx_to_pair(idx) for idx in random_idx]
            util_random, _ = self.lp_solver.solve(tm, random_flows)
            results["random_k"].append(util_random)
        
        summary = {}
        for method, utils in results.items():
            summary[method] = {
                "mean": np.mean(utils),
                "std": np.std(utils),
            }
        return summary
    
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Model loaded from {path}")


def test_trainer():
    """Test the trainer."""
    topo = Topology(n_edge=4, hosts_per_edge=4)
    trainer = REINFORCETrainer(topo, k_critical=8, lr=1e-3, entropy_coef=0.05)
    
    traffic_params = {
        "n_elephant": 8,
        "elephant_demand": 600e6,
        "mice_demand": 60e6,
        "sparsity": 0.5,
    }
    
    print("=" * 60)
    print("BEFORE TRAINING")
    print("=" * 60)
    results_before = trainer.evaluate(n_episodes=20, **traffic_params)
    for method in ["single_path", "random_k", "top_k", "cfr_rl", "ecmp"]:
        print(f"{method:<15} {results_before[method]['mean']:.4f}")
    
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    trainer.train(
        n_iterations=500,
        pretrain=True,
        pretrain_iterations=300,
        log_interval=100,
        **traffic_params
    )
    
    print("\n" + "=" * 60)
    print("AFTER TRAINING")
    print("=" * 60)
    results_after = trainer.evaluate(n_episodes=30, **traffic_params)
    
    single = results_after["single_path"]["mean"]
    ecmp = results_after["ecmp"]["mean"]
    
    print(f"{'Method':<15} {'Util':>8} {'vs Single':>12} {'ECMP Gap':>10}")
    print("-" * 50)
    for method in ["single_path", "random_k", "top_k", "cfr_rl", "ecmp"]:
        util = results_after[method]["mean"]
        vs_single = (single - util) / single * 100
        if method in ["single_path", "ecmp"]:
            gap = "-"
        else:
            gap = f"{(single - util) / (single - ecmp) * 100:.0f}%"
        print(f"{method:<15} {util:>8.4f} {vs_single:>+11.1f}% {gap:>10}")
    
    cfr = results_after["cfr_rl"]["mean"]
    topk = results_after["top_k"]["mean"]
    print(f"\n{'✓ CFR-RL beats Top-K!' if cfr < topk else '~ CFR-RL matches Top-K' if abs(cfr-topk)/topk < 0.05 else '✗ Top-K still better'}")


if __name__ == "__main__":
    test_trainer()