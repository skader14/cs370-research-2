"""
CFR-RL Trainer for Abilene - LATENCY-AWARE VERSION

============================================================================
WHAT THIS FILE DOES
============================================================================

This is a modified version of the trainer that supports latency-aware training.
The key changes are:

1. TrainingConfig has a new field: latency_weight
2. train_step() passes latency_weight to solver.solve()
3. evaluate() passes latency_weight to solver.solve()

The model architecture and training loop are unchanged - we're only changing
what objective the LP solver optimizes for.

============================================================================
USAGE
============================================================================

# Train with MLU-only objective (original paper)
config = TrainingConfig(latency_weight=0.0)
trainer = AbileneRLTrainerLatencyAware(config)
trainer.train()

# Train with latency-aware objective  
config = TrainingConfig(latency_weight=0.3)
trainer = AbileneRLTrainerLatencyAware(config)
trainer.train()

# Run comparison experiment
python abilene_trainer_latency.py --compare

============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import time
import json
import csv
from datetime import datetime
from pathlib import Path

from abilene_topology import AbileneToplogy
from abilene_lp_solver_latency import AbileneLPSolverLatencyAware
from abilene_traffic_gen import AbileneTrafficGenerator

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CFRRLTrainer")


# =============================================================================
# MODEL (unchanged from v2)
# =============================================================================

class StablePointwisePolicy(nn.Module):
    """
    Pointwise scoring network.
    
    No changes from v2 - the model architecture doesn't need to change
    for latency-aware training. We're only changing the LP objective.
    """
    
    def __init__(self, num_flows: int, hidden_dim: int = 64):
        super().__init__()
        self.num_flows = num_flows
        
        self.flow_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        self.global_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
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
        scores = torch.clamp(scores, -10, 10)
        
        return scores
    
    def select_flows(self, flow_features: np.ndarray, k: int,
                    deterministic: bool = False,
                    temperature: float = 1.0,
                    mask: np.ndarray = None) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
        """Select k flows with numerical stability."""
        x = torch.FloatTensor(flow_features).unsqueeze(0)
        scores = self.forward(x).squeeze(0)
        
        if mask is not None:
            mask_tensor = torch.FloatTensor(mask)
            scores = scores * mask_tensor + (1 - mask_tensor) * (-100)
        
        if deterministic:
            selected = torch.topk(scores, k).indices.tolist()
            log_prob = torch.tensor(0.0)
            entropy = torch.tensor(0.0)
        else:
            scaled_scores = scores / max(temperature, 0.1)
            scaled_scores = scaled_scores - scaled_scores.max()
            
            probs = torch.softmax(scaled_scores, dim=0)
            probs = probs.clamp(min=1e-8, max=1.0)
            probs = probs / probs.sum()
            
            try:
                selected = torch.multinomial(probs, k, replacement=False).tolist()
            except RuntimeError:
                selected = torch.topk(scores, k).indices.tolist()
            
            log_probs = torch.log(probs + 1e-10)
            log_prob = log_probs[selected].sum()
            entropy = -(probs * log_probs).sum()
            
            if torch.isnan(entropy) or torch.isnan(log_prob):
                entropy = torch.tensor(0.0)
                log_prob = torch.tensor(0.0)
        
        return selected, log_prob, entropy


# =============================================================================
# TRAINING CONFIG
# =============================================================================

@dataclass
class TrainingConfig:
    """
    Training configuration.
    
    NEW FIELD: latency_weight
    --------------------------
    This is the key addition for latency-aware training.
    - 0.0 = Pure MLU optimization (original paper)
    - >0  = Latency-aware (higher = more latency focus)
    """
    # Model
    k_critical: int = 8
    hidden_dim: int = 64
    
    # Training
    total_iterations: int = 5000
    lr: float = 1e-3
    entropy_coef: float = 0.02
    grad_clip: float = 1.0
    
    # Temperature schedule
    temperature: float = 1.0
    temperature_decay: float = 0.999
    min_temperature: float = 0.3
    
    # Evaluation
    eval_interval: int = 500
    eval_episodes: int = 50
    log_interval: int = 100
    
    # Early stopping
    early_stop_patience: int = 3
    
    # =========================================================================
    # NEW: LATENCY-AWARE PARAMETER
    # =========================================================================
    latency_weight: float = 0.0  # 0 = MLU only, >0 = latency-aware
    
    def to_dict(self):
        return asdict(self)


# =============================================================================
# TRAINER
# =============================================================================

class AbileneRLTrainerLatencyAware:
    """
    RL trainer with latency-aware objective support.
    
    KEY CHANGES FROM V2:
    --------------------
    1. Passes latency_weight to solver.solve() in train_step()
    2. Passes latency_weight to solver.solve() in evaluate()
    3. Logs latency_weight in training output
    4. Saves latency_weight in model checkpoint
    
    Everything else is the same - model, training loop, etc.
    """
    
    def __init__(self, config: TrainingConfig = None, seed: int = 42):
        self.config = config or TrainingConfig()
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.topo = AbileneToplogy()
        self.solver = AbileneLPSolverLatencyAware(self.topo, verbose=False)
        self.traffic_gen = AbileneTrafficGenerator(self.topo, seed=seed)
        
        self.policy = StablePointwisePolicy(
            num_flows=self.topo.num_flows,
            hidden_dim=self.config.hidden_dim
        )
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.config.lr)
        
        self._compute_flow_metadata()
        
        # Training state
        self.history = []
        self.best_performance = float('inf')
        self.best_vs_topk = float('inf')
        self.no_improvement_count = 0
        self.current_temperature = self.config.temperature
        
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Log the latency weight prominently
        logger.info(f"="*60)
        logger.info(f"Trainer initialized")
        logger.info(f"  latency_weight = {self.config.latency_weight}")
        logger.info(f"  (0 = MLU only, >0 = latency aware)")
        logger.info(f"="*60)
        
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
        """
        Single training step.
        
        KEY CHANGE: Passes self.config.latency_weight to solver.solve()
        """
        # Generate traffic
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
        
        # Select flows
        selected, log_prob, entropy = self.policy.select_flows(
            features,
            self.config.k_critical,
            deterministic=False,
            temperature=self.current_temperature,
            mask=self.is_multi_path
        )
        
        # =================================================================
        # SOLVE LP WITH LATENCY WEIGHT (KEY CHANGE)
        # =================================================================
        util_policy, routing = self.solver.solve(
            tm, selected,
            background_routing="single_path",
            latency_weight=self.config.latency_weight  # <-- PASSES LATENCY WEIGHT
        )
        
        if util_policy == float('inf'):
            return {'skipped': True}
        
        # Get baseline (Top-K) with same latency weight for fair comparison
        top_k_flows = self.get_top_k_indices(tm)
        util_top_k, _ = self.solver.solve(
            tm, top_k_flows,
            background_routing="single_path",
            latency_weight=self.config.latency_weight  # <-- SAME WEIGHT
        )
        
        util_single = self.solver.solve_single_path(tm)
        
        # Reward (higher is better)
        reward = 1.0 / util_policy
        baseline = 1.0 / util_top_k if util_top_k > 0 else reward
        advantage = reward - baseline
        
        # Policy gradient
        self.optimizer.zero_grad()
        loss = -log_prob * advantage - self.config.entropy_coef * entropy
        
        if not torch.isnan(loss):
            loss.backward()
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
            'latency_weight': self.config.latency_weight,
            'pattern': pattern,
            'skipped': False
        }
    
    def evaluate(self, n_episodes: int = None) -> Dict[str, float]:
        """
        Evaluate current policy.
        
        KEY CHANGE: Passes self.config.latency_weight to solver.solve()
        """
        n_episodes = n_episodes or self.config.eval_episodes
        
        results = {
            'single_path': [],
            'random_k': [],
            'top_k': [],
            'cfr_rl': [],
            'ecmp': []
        }
        
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
            
            # Random-K baseline
            util_random, _ = self.solver.solve_random_k(
                tm, self.config.k_critical, 
                latency_weight=self.config.latency_weight)
            results['random_k'].append(util_random)
            
            # Top-K baseline
            util_top_k, _ = self.solver.solve_top_k(
                tm, self.config.k_critical, 
                latency_weight=self.config.latency_weight)
            results['top_k'].append(util_top_k)
            
            # CFR-RL selection
            selected, _, _ = self.policy.select_flows(
                features, self.config.k_critical,
                deterministic=True,
                mask=self.is_multi_path
            )
            
            # =============================================================
            # SOLVE WITH LATENCY WEIGHT (KEY CHANGE)
            # =============================================================
            util_cfr, _ = self.solver.solve(
                tm, selected, 
                background_routing="single_path",
                latency_weight=self.config.latency_weight  # <-- PASSES WEIGHT
            )
            
            if util_cfr == float('inf'):
                util_cfr = util_top_k
            
            results['cfr_rl'].append(util_cfr)
            
            if util_cfr <= util_top_k * 1.001:
                cfr_wins += 1
        
        avg_results = {k: np.mean(v) for k, v in results.items()}
        avg_results['win_rate'] = cfr_wins / n_episodes * 100
        
        return avg_results
    
    def _print_eval(self, results: Dict[str, float], iteration: int) -> bool:
        """Print eval results and return True if improved."""
        print(f"\n  --- Eval @ iter {iteration} (lw={self.config.latency_weight}) ---")
        
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
                model_name = f"best_abilene_lw{self.config.latency_weight}.pt"
                print(f"    ★ New best! Saving to {model_name}")
                self.save(model_name)
        else:
            gap = (cfr - topk) / topk * 100
            print(f"  ✗ Top-K beats CFR-RL by {gap:.1f}%")
        
        print()
        return improved
    
    def train(self) -> List[Dict]:
        """Main training loop."""
        print(f"\n{'='*60}")
        print(f"CFR-RL TRAINING")
        print(f"{'='*60}")
        print(f"latency_weight = {self.config.latency_weight}")
        print(f"  (0 = MLU only, >0 = latency aware)")
        print(f"iterations = {self.config.total_iterations}")
        print(f"k_critical = {self.config.k_critical}")
        
        for i in range(self.config.total_iterations):
            metrics = self.train_step()
            
            if not metrics.get('skipped', False):
                self.history.append(metrics)
            
            if (i + 1) % self.config.log_interval == 0:
                recent = [h for h in self.history[-self.config.log_interval:] if not h.get('skipped')]
                if recent:
                    avg_policy = np.mean([m['util_policy'] for m in recent])
                    avg_topk = np.mean([m['util_top_k'] for m in recent])
                    avg_entropy = np.mean([m['entropy'] for m in recent])
                    avg_overlap = np.mean([m['overlap'] for m in recent])
                    temp = self.current_temperature
                    
                    wins = sum(1 for m in recent if m['util_policy'] <= m['util_top_k'] * 1.01)
                    win_rate = wins / len(recent) * 100
                    
                    status = "✓" if avg_policy <= avg_topk else "✗"
                    print(f"  Iter {i+1:5d} | Policy: {avg_policy:.3f} | Top-K: {avg_topk:.3f} | "
                          f"Win: {win_rate:4.0f}% | Overlap: {avg_overlap:.1f}/{self.config.k_critical} | "
                          f"Ent: {avg_entropy:.2f} | T: {temp:.2f} {status}")
            
            if (i + 1) % self.config.eval_interval == 0:
                eval_results = self.evaluate()
                improved = self._print_eval(eval_results, i + 1)
                
                if improved:
                    self.no_improvement_count = 0
                else:
                    self.no_improvement_count += 1
                
                if self.no_improvement_count >= self.config.early_stop_patience:
                    print(f"\n  Early stopping: no improvement for {self.config.early_stop_patience} evals")
                    break
        
        return self.history
    
    def save(self, path: str):
        """Save model with latency_weight in checkpoint."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'config': self.config.to_dict(),
            'best_vs_topk': self.best_vs_topk,
            'latency_weight': self.config.latency_weight,  # <-- SAVE WEIGHT
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, weights_only=False)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        return checkpoint


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='CFR-RL Latency-Aware Training')
    parser.add_argument('--latency-weight', type=float, default=0.0,
                       help='Latency weight (0=MLU only, >0=latency-aware)')
    parser.add_argument('--iterations', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple latency weights')
    
    args = parser.parse_args()
    
    if args.compare:
        # Run comparison across latency weights
        print("="*70)
        print("COMPARISON EXPERIMENT")
        print("="*70)
        
        results = {}
        for lw in [0.0, 0.1, 0.3, 0.5]:
            print(f"\n{'='*70}")
            print(f"TRAINING WITH latency_weight={lw}")
            print(f"{'='*70}")
            
            config = TrainingConfig(
                latency_weight=lw,
                total_iterations=args.iterations,
            )
            
            trainer = AbileneRLTrainerLatencyAware(config, seed=args.seed)
            trainer.train()
            
            # Load best and evaluate
            try:
                trainer.load(f"best_abilene_lw{lw}.pt")
            except:
                pass
            
            final = trainer.evaluate(n_episodes=100)
            results[lw] = final
        
        # Print summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"{'Weight':<10} {'MLU':<10} {'Win Rate':<10}")
        for lw, r in results.items():
            print(f"{lw:<10} {r['cfr_rl']:<10.4f} {r['win_rate']:<10.0f}%")
    
    else:
        # Single training run
        config = TrainingConfig(
            latency_weight=args.latency_weight,
            total_iterations=args.iterations,
        )
        
        trainer = AbileneRLTrainerLatencyAware(config, seed=args.seed)
        
        print("\n" + "="*60)
        print("BEFORE TRAINING")
        print("="*60)
        results_before = trainer.evaluate()
        trainer._print_eval(results_before, 0)
        
        trainer.train()

    # Load best model for final evaluation
    model_path = f"best_abilene_lw{args.latency_weight}.pt"
    try:
        trainer.load(model_path)
        print(f"\nLoaded best model from {model_path}")
    except Exception as e:
        print(f"\nWARNING: Could not load best model: {e}")
        print("Using final training state instead")

    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    results_after = trainer.evaluate(n_episodes=100)
    trainer._print_eval(results_after, -1)


if __name__ == "__main__":
    main()