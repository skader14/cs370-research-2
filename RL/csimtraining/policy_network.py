"""
CFR-RL Policy Network - FIXED VERSION

Key fixes:
1. Proper weight initialization (Kaiming for ReLU)
2. Gradient clipping to prevent explosion
3. Entropy regularization to prevent collapse
4. Simplified log_prob computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import List, Tuple, Optional, Dict
import numpy as np
from collections import deque


class PolicyNetwork(nn.Module):
    """
    Policy network for CFR-RL critical flow selection.
    
    Architecture: MLP that maps flow features to per-flow scores.
    Action: Select top-k flows based on scores (with Gumbel noise for exploration).
    """
    
    def __init__(
        self,
        num_flows: int = 240,
        num_features: int = 9,
        hidden_dim: int = 256,
        num_hidden_layers: int = 2,
    ):
        super().__init__()
        
        self.num_flows = num_flows
        self.num_features = num_features
        input_dim = num_flows * num_features
        
        # Build network
        layers = []
        in_dim = input_dim
        
        for i in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))  # Added LayerNorm for stability
            in_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        self.score_head = nn.Linear(hidden_dim, num_flows)
        
        self._init_weights()
    
    def _init_weights(self):
        """Proper initialization for ReLU networks."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Kaiming init for ReLU (gain = sqrt(2))
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Output layer with smaller weights for stable initial policy
        nn.init.xavier_normal_(self.score_head.weight, gain=0.1)
        nn.init.zeros_(self.score_head.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: features -> per-flow scores."""
        if x.dim() == 3:
            x = x.view(x.size(0), -1)
        
        features = self.feature_extractor(x)
        scores = self.score_head(features)
        return scores
    
    def sample_action(
        self,
        scores: torch.Tensor,
        k: int,
        temperature: float = 1.0,
    ) -> Tuple[List[int], torch.Tensor]:
        """
        Sample K flows using Gumbel-Softmax.
        
        Returns:
            selected_flows: List of selected flow indices
            log_prob: Log probability of this selection
        """
        if scores.dim() == 2:
            scores = scores.squeeze(0)
        
        # Temperature scaling
        scaled_scores = scores / temperature
        
        # Gumbel-Softmax sampling
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(scaled_scores) + 1e-10) + 1e-10)
        perturbed_scores = scaled_scores + gumbel_noise
        
        # Select top-k
        _, indices = torch.topk(perturbed_scores, k)
        selected_flows = indices.tolist()
        
        # Log probability (sum of log-softmax for selected flows)
        log_probs = F.log_softmax(scaled_scores, dim=0)
        log_prob = log_probs[indices].sum()
        
        return selected_flows, log_prob
    
    def evaluate_action(
        self,
        scores: torch.Tensor,
        selected_flows: List[int],
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Compute log probability of a given action."""
        if scores.dim() == 2:
            scores = scores.squeeze(0)
        
        scaled_scores = scores / temperature
        log_probs = F.log_softmax(scaled_scores, dim=0)
        
        indices = torch.tensor(selected_flows, device=scores.device)
        return log_probs[indices].sum()
    
    def get_entropy(self, scores: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Compute entropy of the score distribution."""
        if scores.dim() == 2:
            scores = scores.squeeze(0)
        probs = F.softmax(scores / temperature, dim=0)
        return -(probs * torch.log(probs + 1e-10)).sum()


class BatchReinforceTrainer:
    """
    REINFORCE trainer with batch updates and proper gradient handling.
    
    Key features:
    1. Batch updates for variance reduction
    2. Advantage normalization
    3. Entropy bonus to prevent collapse
    4. Gradient clipping
    """
    
    def __init__(
        self,
        policy: nn.Module,
        lr: float = 1e-3,
        batch_size: int = 10,
        num_updates_per_batch: int = 1,  # Reduced from 3 - one update is often better
        entropy_weight: float = 0.01,
        max_grad_norm: float = 1.0,
        normalize_advantages: bool = True,
        lr_schedule: bool = False,  # Disabled by default - can hurt learning
        total_episodes: int = 500,
    ):
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        
        if lr_schedule:
            self.scheduler = CosineAnnealingLR(
                self.optimizer, 
                T_max=max(total_episodes // batch_size, 1)
            )
        else:
            self.scheduler = None
        
        self.batch_size = batch_size
        self.num_updates_per_batch = num_updates_per_batch
        self.entropy_weight = entropy_weight
        self.max_grad_norm = max_grad_norm
        self.normalize_advantages = normalize_advantages
        
        self.episode_buffer = []
        self.reward_history = deque(maxlen=100)
        self.total_updates = 0
    
    def store_episode(
        self,
        features: torch.Tensor,
        selected_flows: List[int],
        log_prob: torch.Tensor,
        reward: float,
        temperature: float,
    ):
        """Store episode for batch update."""
        self.episode_buffer.append({
            'features': features.detach().clone(),  # Clone to be safe
            'selected_flows': list(selected_flows),  # Copy the list
            'reward': float(reward),
            'temperature': float(temperature),
        })
        self.reward_history.append(reward)
    
    def should_update(self) -> bool:
        """Check if ready for batch update."""
        return len(self.episode_buffer) >= self.batch_size
    
    def update(self) -> Dict[str, float]:
        """Perform batch REINFORCE update."""
        if len(self.episode_buffer) < self.batch_size:
            return {'loss': 0.0, 'skipped': True}
        
        # Get batch
        batch = self.episode_buffer[:self.batch_size]
        self.episode_buffer = self.episode_buffer[self.batch_size:]
        
        # Compute advantages
        rewards = torch.tensor([ep['reward'] for ep in batch], dtype=torch.float32)
        baseline = rewards.mean()
        advantages = rewards - baseline
        
        # Normalize advantages
        if self.normalize_advantages and advantages.std() > 1e-8:
            advantages = advantages / (advantages.std() + 1e-8)
        
        # Gradient update(s)
        total_loss = 0.0
        total_policy_loss = 0.0
        total_entropy = 0.0
        
        for _ in range(self.num_updates_per_batch):
            self.optimizer.zero_grad()
            
            batch_policy_loss = []
            batch_entropy = []
            
            for i, ep in enumerate(batch):
                # Forward pass with gradient
                scores = self.policy(ep['features'])
                
                # Log prob for this action
                log_prob = self.policy.evaluate_action(
                    scores, 
                    ep['selected_flows'], 
                    ep['temperature']
                )
                
                # Policy gradient loss
                policy_loss = -advantages[i] * log_prob
                batch_policy_loss.append(policy_loss)
                
                # Entropy
                entropy = self.policy.get_entropy(scores, ep['temperature'])
                batch_entropy.append(entropy)
            
            # Average over batch
            mean_policy_loss = torch.stack(batch_policy_loss).mean()
            mean_entropy = torch.stack(batch_entropy).mean()
            
            # Total loss (subtract entropy to encourage exploration)
            loss = mean_policy_loss - self.entropy_weight * mean_entropy
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), 
                self.max_grad_norm
            )
            
            # Update
            self.optimizer.step()
            
            total_loss += loss.item()
            total_policy_loss += mean_policy_loss.item()
            total_entropy += mean_entropy.item()
        
        # LR scheduling
        if self.scheduler is not None:
            self.scheduler.step()
        
        self.total_updates += 1
        
        return {
            'loss': total_loss / self.num_updates_per_batch,
            'policy_loss': total_policy_loss / self.num_updates_per_batch,
            'entropy': total_entropy / self.num_updates_per_batch,
            'baseline': baseline.item(),
            'batch_reward_mean': rewards.mean().item(),
            'batch_reward_std': rewards.std().item(),
            'advantage_std': advantages.std().item() if advantages.std() > 0 else 0,
            'lr': self.optimizer.param_groups[0]['lr'],
        }
    
    def get_baseline(self) -> float:
        """Get running baseline."""
        if len(self.reward_history) > 0:
            return float(np.mean(list(self.reward_history)))
        return 0.0


# =============================================================================
# Quick test
# =============================================================================

if __name__ == "__main__":
    print("Testing PolicyNetwork and BatchReinforceTrainer...")
    
    # Create network
    policy = PolicyNetwork(num_flows=50, num_features=5)
    trainer = BatchReinforceTrainer(policy, lr=0.01, batch_size=5)
    
    NUM_EPISODES = 100
    K = 5
    rewards = []
    
    for ep in range(NUM_EPISODES):
        # Random features
        features = torch.randn(1, 50 * 5)
        
        # Sample action
        scores = policy(features)
        selected, log_prob = policy.sample_action(scores, k=K, temperature=1.0)
        
        # Synthetic reward: sum of selected indices (higher is better)
        reward = sum(selected) / 50.0
        rewards.append(reward)
        
        # Store and update
        trainer.store_episode(features, selected, log_prob, reward, 1.0)
        
        if trainer.should_update():
            metrics = trainer.update()
            if ep % 20 == 0:
                print(f"Ep {ep}: reward={reward:.3f}, avg={np.mean(rewards[-20:]):.3f}")
    
    print(f"\nFirst 20 avg: {np.mean(rewards[:20]):.3f}")
    print(f"Last 20 avg:  {np.mean(rewards[-20:]):.3f}")
    
    if np.mean(rewards[-20:]) > np.mean(rewards[:20]) + 0.05:
        print("✓ Learning detected!")
    else:
        print("✗ No clear learning")