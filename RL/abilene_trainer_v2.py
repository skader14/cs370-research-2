"""
CFR-RL Trainer for Abilene - V2 (Fixed)

Fixes from V1:
1. Higher minimum temperature (0.3 instead of 0.1)
2. Numerical stability in softmax/log
3. Early stopping when we find a good model
4. Gradient norm monitoring
5. Model saved when CFR-RL beats Top-K
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


class StablePointwisePolicy(nn.Module):
    """
    Pointwise scoring with numerical stability.
    """
    
    def __init__(self, num_flows: int, hidden_dim: int = 64):
        super().__init__()
        self.num_flows = num_flows
        
        # Per-flow encoder
        self.flow_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Added for stability
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Added for stability
            nn.ReLU(),
        )
        
        # Global context
        self.global_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Score head
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights carefully
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, flow_features: torch.Tensor) -> torch.Tensor:
        batch_size = flow_features.shape[0]
        
        local_encoded = self.flow_encoder(flow_features)
        global_context = local_encoded.mean(dim=1, keepdim=True)
        global_context = self.global_encoder(global_context)
        global_context = global_context.expand(-1, self.num_flows, -1)
        
        combined = torch.cat([local_encoded, global_context], dim=-1)
        scores = self.score_head(combined).squeeze(-1)
        
        # Clamp scores for stability
        scores = torch.clamp(scores, -10, 10)
        
        return scores
    
    def select_flows(self, flow_features: np.ndarray, k: int,
                    deterministic: bool = False,
                    temperature: float = 1.0,
                    mask: np.ndarray = None) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
        """Select k flows with numerical stability."""
        x = torch.FloatTensor(flow_features).unsqueeze(0)
        scores = self.forward(x).squeeze(0)
        
        # Apply mask
        if mask is not None:
            mask_tensor = torch.FloatTensor(mask)
            scores = scores * mask_tensor + (1 - mask_tensor) * (-100)
        
        if deterministic:
            selected = torch.topk(scores, k).indices.tolist()
            log_prob = torch.tensor(0.0)
            entropy = torch.tensor(0.0)
        else:
            # Stable softmax computation
            scaled_scores = scores / max(temperature, 0.1)
            scaled_scores = scaled_scores - scaled_scores.max()  # Numerical stability
            
            probs = torch.softmax(scaled_scores, dim=0)
            probs = probs.clamp(min=1e-8, max=1.0)  # Prevent exact 0s
            probs = probs / probs.sum()  # Renormalize
            
            # Sample
            try:
                selected = torch.multinomial(probs, k, replacement=False).tolist()
            except RuntimeError:
                # Fallback to top-k
                selected = torch.topk(scores, k).indices.tolist()
            
            # Stable log probability
            log_probs = torch.log(probs + 1e-10)
            log_prob = log_probs[selected].sum()
            
            # Entropy
            entropy = -(probs * log_probs).sum()
            
            # Check for NaN
            if torch.isnan(entropy) or torch.isnan(log_prob):
                entropy = torch.tensor(0.0)
                log_prob = torch.tensor(0.0)
        
        return selected, log_prob, entropy


@dataclass
class TrainingConfig:
    k_critical: int = 8
    hidden_dim: int = 64
    
    total_iterations: int = 5000  # Reduced - we'll early stop
    lr: float = 1e-3
    entropy_coef: float = 0.02
    grad_clip: float = 1.0
    
    # Temperature schedule - don't go too low
    temperature: float = 1.0
    temperature_decay: float = 0.999
    min_temperature: float = 0.3  # Higher minimum!
    
    eval_interval: int = 500
    eval_episodes: int = 50
    log_interval: int = 100
    
    # Early stopping
    early_stop_patience: int = 3  # Stop after 3 evals without improvement


class AbileneRLTrainerV2:
    """Fixed RL trainer with stability improvements."""
    
    def __init__(self, config: TrainingConfig = None, seed: int = 42):
        self.config = config or TrainingConfig()
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.topo = AbileneToplogy()
        self.solver = AbileneLPSolver(self.topo)
        self.traffic_gen = AbileneTrafficGenerator(self.topo, seed=seed)
        
        self.policy = StablePointwisePolicy(
            num_flows=self.topo.num_flows,
            hidden_dim=self.config.hidden_dim
        )
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.config.lr)
        
        self._compute_flow_metadata()
        
        self.history = []
        self.best_performance = float('inf')
        self.best_vs_topk = float('inf')  # Track best CFR-RL/Top-K ratio
        self.no_improvement_count = 0
        self.current_temperature = self.config.temperature
        
    def _compute_flow_metadata(self):
        self.num_paths = np.zeros(self.topo.num_flows)
        self.path_lengths = np.zeros(self.topo.num_flows)
        self.bottleneck_caps = np.zeros(self.topo.num_flows)
        self.is_multi_path = np.zeros(self.topo.num_flows)
        
        for flow_idx, (src, dst) in enumerate(self.topo.flow_pairs):
            paths = self.topo.get_all_paths(src, dst)
            self.num_paths[flow_idx] = len(paths)
            self.is_multi_path[flow_idx] = float(len(paths) > 1)
            
            if paths:
                self.path_lengths[flow_idx] = len(paths[0]) - 1
                self.bottleneck_caps[flow_idx] = self.topo.get_path_bottleneck(paths[0])
    
    def prepare_features(self, tm: np.ndarray) -> np.ndarray:
        flows = self.traffic_gen.tm_to_flow_vector(tm)
        
        max_flow = flows.max() if flows.max() > 0 else 1.0
        max_cap = self.bottleneck_caps.max() if self.bottleneck_caps.max() > 0 else 1.0
        
        features = np.stack([
            flows / max_flow,
            self.num_paths / 4.0,
            self.path_lengths / 5.0,
            self.bottleneck_caps / max_cap,
        ], axis=1)
        
        return features
    
    def get_top_k_indices(self, tm: np.ndarray) -> List[int]:
        flows = self.traffic_gen.tm_to_flow_vector(tm)
        masked = flows * self.is_multi_path
        return np.argsort(-masked)[:self.config.k_critical].tolist()
    
    def train_step(self) -> Dict:
        # Diverse traffic
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
        
        # Policy selection
        selected, log_prob, entropy = self.policy.select_flows(
            features,
            self.config.k_critical,
            deterministic=False,
            temperature=self.current_temperature,
            mask=self.is_multi_path
        )
        
        # Evaluate
        util_policy, _ = self.solver.solve(tm, selected, background_routing="single_path")
        
        # Check for solver failure
        if util_policy == float('inf') or util_policy > 100:
            # Skip this step
            return {
                'util_policy': util_top_k,  # Use top-k as fallback
                'util_top_k': util_top_k,
                'util_single': util_single,
                'entropy': entropy.item() if not torch.isnan(entropy) else 0.0,
                'overlap': 0,
                'temperature': self.current_temperature,
                'skipped': True
            }
        
        # Reward: improvement over single-path
        reward = (util_single - util_policy) / max(util_single, 0.01)
        
        # Baseline: running average
        baseline = getattr(self, 'reward_baseline', reward)
        advantage = reward - baseline
        self.reward_baseline = 0.99 * baseline + 0.01 * reward
        
        # Loss
        if not torch.isnan(log_prob) and not torch.isnan(entropy):
            policy_loss = -log_prob * advantage
            entropy_loss = -self.config.entropy_coef * entropy
            loss = policy_loss + entropy_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            
            # Monitor gradient norm
            grad_norm = 0.0
            for p in self.policy.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            
            # Only update if gradients are reasonable
            if grad_norm < 100 and not np.isnan(grad_norm):
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.grad_clip)
                self.optimizer.step()
        
        # Temperature decay
        self.current_temperature = max(
            self.config.min_temperature,
            self.current_temperature * self.config.temperature_decay
        )
        
        overlap = len(set(selected) & set(top_k_flows))
        
        return {
            'util_policy': util_policy,
            'util_top_k': util_top_k,
            'util_single': util_single,
            'entropy': entropy.item() if not torch.isnan(entropy) else 0.0,
            'overlap': overlap,
            'temperature': self.current_temperature,
            'skipped': False
        }
    
    def evaluate(self, n_episodes: int = None) -> Dict[str, float]:
        n_episodes = n_episodes or self.config.eval_episodes
        
        results = {
            'single_path': [],
            'random_k': [],
            'top_k': [],
            'cfr_rl': [],
            'ecmp': []
        }
        
        # Track individual comparisons
        cfr_wins = 0
        
        for _ in range(n_episodes):
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
            
            selected, _, _ = self.policy.select_flows(
                features, self.config.k_critical,
                deterministic=True,
                mask=self.is_multi_path
            )
            util_cfr, _ = self.solver.solve(tm, selected, background_routing="single_path")
            
            # Handle solver failure
            if util_cfr == float('inf'):
                util_cfr = util_top_k
            
            results['cfr_rl'].append(util_cfr)
            
            if util_cfr <= util_top_k * 1.001:  # Allow tiny tolerance
                cfr_wins += 1
        
        avg_results = {k: np.mean(v) for k, v in results.items()}
        avg_results['win_rate'] = cfr_wins / n_episodes * 100
        
        return avg_results
    
    def _print_eval(self, results: Dict[str, float], iteration: int) -> bool:
        """Print eval and return True if improved."""
        print(f"\n  --- Eval @ iter {iteration} ---")
        
        single = results['single_path']
        cfr = results['cfr_rl']
        topk = results['top_k']
        
        for method in ['single_path', 'random_k', 'top_k', 'cfr_rl', 'ecmp']:
            util = results[method]
            vs_single = (single - util) / single * 100
            print(f"    {method:<12}: {util:.4f} ({vs_single:+.1f}% vs single)")
        
        print(f"    Win rate: {results['win_rate']:.0f}%")
        
        ratio = cfr / topk
        improved = False
        
        if cfr < topk:
            improvement = (topk - cfr) / topk * 100
            print(f"  ✓ CFR-RL beats Top-K by {improvement:.1f}%")
            if ratio < self.best_vs_topk:
                self.best_vs_topk = ratio
                improved = True
                print(f"    ★ New best! Saving model...")
                self.save("best_abilene_v2.pt")
        else:
            gap = (cfr - topk) / topk * 100
            print(f"  ✗ Top-K beats CFR-RL by {gap:.1f}%")
        
        print()
        return improved
    
    def train(self) -> List[Dict]:
        print(f"\n{'='*60}")
        print("CFR-RL TRAINING (STABLE VERSION)")
        print(f"{'='*60}")
        print(f"Iterations: {self.config.total_iterations}")
        print(f"K critical: {self.config.k_critical}")
        print(f"Temperature: {self.config.temperature} -> {self.config.min_temperature}")
        print(f"Early stop patience: {self.config.early_stop_patience}")
        
        start_time = time.time()
        
        for i in range(self.config.total_iterations):
            metrics = self.train_step()
            
            if not metrics.get('skipped', False):
                self.history.append(metrics)
            
            # Logging
            if (i + 1) % self.config.log_interval == 0:
                recent = [h for h in self.history[-self.config.log_interval:] if not h.get('skipped')]
                if recent:
                    avg_policy = np.mean([m['util_policy'] for m in recent])
                    avg_topk = np.mean([m['util_top_k'] for m in recent])
                    avg_entropy = np.mean([m['entropy'] for m in recent])
                    avg_overlap = np.mean([m['overlap'] for m in recent])
                    temp = metrics['temperature']
                    
                    wins = sum(1 for m in recent if m['util_policy'] <= m['util_top_k'] * 1.01)
                    win_rate = wins / len(recent) * 100
                    
                    status = "✓" if avg_policy <= avg_topk else "✗"
                    print(f"  Iter {i+1:5d} | Policy: {avg_policy:.3f} | Top-K: {avg_topk:.3f} | "
                          f"Win: {win_rate:4.0f}% | Overlap: {avg_overlap:.1f}/{self.config.k_critical} | "
                          f"Ent: {avg_entropy:.2f} | T: {temp:.2f} {status}")
            
            # Evaluation
            if (i + 1) % self.config.eval_interval == 0:
                eval_results = self.evaluate()
                improved = self._print_eval(eval_results, i + 1)
                
                if improved:
                    self.no_improvement_count = 0
                else:
                    self.no_improvement_count += 1
                
                # Early stopping
                if self.no_improvement_count >= self.config.early_stop_patience:
                    print(f"\n  Early stopping: no improvement for {self.config.early_stop_patience} evals")
                    break
        
        elapsed = time.time() - start_time
        print(f"\nTraining complete in {elapsed/60:.1f} minutes")
        
        return self.history
    
    def save(self, path: str):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'config': self.config,
            'best_vs_topk': self.best_vs_topk,
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        print(f"Loaded model from {path}")


def main():
    print("="*60)
    print("CFR-RL ON ABILENE - STABLE VERSION")
    print("="*60)
    
    config = TrainingConfig(
        k_critical=8,
        hidden_dim=64,
        total_iterations=5000,
        lr=1e-3,
        entropy_coef=0.02,
        temperature=1.0,
        temperature_decay=0.999,
        min_temperature=0.3,  # Don't go below 0.3
        eval_interval=500,
        log_interval=100,
        early_stop_patience=3,
    )
    
    trainer = AbileneRLTrainerV2(config, seed=42)
    trainer.topo.print_summary()
    
    # Before
    print("\n" + "="*60)
    print("BEFORE TRAINING")
    print("="*60)
    results_before = trainer.evaluate()
    trainer._print_eval(results_before, 0)
    
    # Train
    trainer.train()
    
    # Load best model for final eval
    try:
        trainer.load("best_abilene_v2.pt")
        print("\nLoaded best model for final evaluation")
    except:
        print("\nUsing final model for evaluation")
    
    # Final eval
    print("\n" + "="*60)
    print("FINAL EVALUATION (100 episodes)")
    print("="*60)
    results_after = trainer.evaluate(n_episodes=100)
    trainer._print_eval(results_after, -1)
    
    return trainer, results_after


if __name__ == "__main__":
    trainer, results = main()