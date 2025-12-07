"""
policy_network.py - Policy network for CFR-RL with improved training.

Changes from v1:
1. Batch REINFORCE - accumulate N episodes before updating
2. Advantage normalization - reduces variance
3. Multi-step updates - multiple gradient steps per batch
4. Learning rate scheduling - decay over training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import deque


# Fat-Tree k=4 constants
NUM_FLOWS = 240
NUM_FEATURES = 9


# =============================================================================
# Policy Network Architecture
# =============================================================================

class PolicyNetwork(nn.Module):
    """
    Neural network that outputs a score for each flow.
    
    Architecture: MLP with residual connections
    Input: (batch, num_flows * num_features)
    Output: (batch, num_flows) - unnormalized scores
    """
    
    def __init__(
        self,
        num_flows: int = NUM_FLOWS,
        num_features: int = NUM_FEATURES,
        hidden_dims: List[int] = [512, 256, 128],
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_flows = num_flows
        self.num_features = num_features
        self.input_dim = num_flows * num_features
        
        # Build MLP layers
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Output head: score per flow
        self.score_head = nn.Linear(prev_dim, num_flows)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features (batch, num_flows * num_features)
               or (batch, num_flows, num_features)
        
        Returns:
            Scores for each flow (batch, num_flows)
        """
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
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[List[int], torch.Tensor]:
        """
        Sample K flows using Gumbel-Softmax trick.
        
        Returns selected flow indices and log probability.
        """
        if scores.dim() == 2:
            scores = scores.squeeze(0)
        
        # Apply temperature
        scaled_scores = scores / temperature
        
        # Apply mask if provided
        if mask is not None:
            scaled_scores = scaled_scores.masked_fill(~mask, float('-inf'))
        
        # Gumbel-Softmax sampling for differentiable top-k
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(scaled_scores) + 1e-10) + 1e-10)
        perturbed_scores = scaled_scores + gumbel_noise
        
        # Select top-k
        _, indices = torch.topk(perturbed_scores, k)
        selected_flows = indices.tolist()
        
        # Compute log probability
        log_probs = F.log_softmax(scaled_scores, dim=0)
        log_prob = log_probs[indices].sum()
        
        return selected_flows, log_prob
    
    def evaluate_action(
        self,
        scores: torch.Tensor,
        selected_flows: List[int],
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute log probability of a given action (for REINFORCE).
        """
        if scores.dim() == 2:
            scores = scores.squeeze(0)
        
        scaled_scores = scores / temperature
        log_probs = F.log_softmax(scaled_scores, dim=0)
        
        indices = torch.tensor(selected_flows, device=scores.device)
        return log_probs[indices].sum()


# =============================================================================
# Batch REINFORCE Trainer
# =============================================================================

class BatchReinforceTrainer:
    """
    Improved REINFORCE trainer with batch updates.
    
    Key improvements:
    1. Accumulates N episodes before updating (reduces variance)
    2. Normalizes advantages within batch (stabilizes training)
    3. Multiple gradient steps per batch (better sample efficiency)
    4. Learning rate scheduling
    """
    
    def __init__(
        self,
        policy: nn.Module,
        lr: float = 3e-4,
        batch_size: int = 10,           # Episodes per batch
        num_updates_per_batch: int = 3, # Gradient steps per batch
        entropy_weight: float = 0.01,
        max_grad_norm: float = 0.5,
        gamma: float = 1.0,             # Discount (1.0 for episodic)
        normalize_advantages: bool = True,
        lr_schedule: bool = True,
        total_episodes: int = 500,
    ):
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        
        # Learning rate scheduler
        if lr_schedule:
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_episodes // batch_size)
        else:
            self.scheduler = None
        
        self.batch_size = batch_size
        self.num_updates_per_batch = num_updates_per_batch
        self.entropy_weight = entropy_weight
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.normalize_advantages = normalize_advantages
        
        # Experience buffer for batch updates
        self.episode_buffer = []
        
        # Running statistics for baseline
        self.reward_history = deque(maxlen=100)
        self.baseline = 0.0
        
        # Metrics tracking
        self.total_updates = 0
    
    def store_episode(
        self,
        features: torch.Tensor,
        selected_flows: List[int],
        log_prob: torch.Tensor,
        reward: float,
        temperature: float,
    ):
        """
        Store episode experience for batch update.
        
        Args:
            features: Feature tensor used for decision
            selected_flows: Flows that were selected
            log_prob: Log probability of action
            reward: Episode reward
            temperature: Temperature used
        """
        self.episode_buffer.append({
            'features': features.detach(),
            'selected_flows': selected_flows,
            'log_prob': log_prob,
            'reward': reward,
            'temperature': temperature,
        })
        
        self.reward_history.append(reward)
    
    def should_update(self) -> bool:
        """Check if we have enough episodes for a batch update."""
        return len(self.episode_buffer) >= self.batch_size
    
    def update(self) -> Dict[str, float]:
        """
        Perform batch REINFORCE update.
        
        Returns metrics dictionary.
        """
        if len(self.episode_buffer) < self.batch_size:
            return {'loss': 0.0, 'skipped': True}
        
        # Extract batch data
        batch = self.episode_buffer[:self.batch_size]
        self.episode_buffer = self.episode_buffer[self.batch_size:]
        
        rewards = torch.tensor([ep['reward'] for ep in batch])
        
        # Compute advantages with baseline
        baseline = rewards.mean().item()
        advantages = rewards - baseline
        
        # Normalize advantages (key for stable training!)
        if self.normalize_advantages and advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Multiple gradient steps
        total_loss = 0.0
        total_policy_loss = 0.0
        total_entropy = 0.0
        
        for update_step in range(self.num_updates_per_batch):
            policy_losses = []
            entropies = []
            
            for i, ep in enumerate(batch):
                # Recompute log probability (allows gradient flow)
                scores = self.policy(ep['features'])
                log_prob = self.policy.evaluate_action(
                    scores, ep['selected_flows'], ep['temperature']
                )
                
                # Policy gradient loss: -advantage * log_prob
                policy_loss = -advantages[i] * log_prob
                policy_losses.append(policy_loss)
                
                # Entropy bonus
                probs = F.softmax(scores.squeeze() / ep['temperature'], dim=0)
                entropy = -(probs * torch.log(probs + 1e-10)).sum()
                entropies.append(entropy)
            
            # Average losses over batch
            policy_loss = torch.stack(policy_losses).mean()
            entropy = torch.stack(entropies).mean()
            entropy_loss = -self.entropy_weight * entropy
            
            loss = policy_loss + entropy_loss
            
            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.max_grad_norm
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_entropy += entropy.item()
        
        # Learning rate scheduling
        if self.scheduler is not None:
            self.scheduler.step()
        
        self.total_updates += 1
        self.baseline = baseline
        
        return {
            'loss': total_loss / self.num_updates_per_batch,
            'policy_loss': total_policy_loss / self.num_updates_per_batch,
            'entropy': total_entropy / self.num_updates_per_batch,
            'baseline': baseline,
            'batch_reward_mean': rewards.mean().item(),
            'batch_reward_std': rewards.std().item(),
            'advantage_std': advantages.std().item(),
            'lr': self.optimizer.param_groups[0]['lr'],
            'num_updates': self.total_updates,
        }
    
    def get_baseline(self) -> float:
        """Get current baseline estimate."""
        if len(self.reward_history) > 0:
            return np.mean(list(self.reward_history))
        return 0.0


# =============================================================================
# Legacy Single-Episode Trainer (for comparison)
# =============================================================================

class ReinforceTrainer:
    """
    Original single-episode REINFORCE trainer.
    Kept for backward compatibility and A/B testing.
    """
    
    def __init__(
        self,
        policy: nn.Module,
        lr: float = 1e-4,
        baseline_decay: float = 0.99,
        entropy_weight: float = 0.01,
        max_grad_norm: float = 1.0,
    ):
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        
        self.baseline_decay = baseline_decay
        self.entropy_weight = entropy_weight
        self.max_grad_norm = max_grad_norm
        
        self.baseline = 0.0
        self.baseline_initialized = False
    
    def update(self, reward: float, log_prob: torch.Tensor) -> float:
        """Single-episode REINFORCE update."""
        # Update baseline
        if not self.baseline_initialized:
            self.baseline = reward
            self.baseline_initialized = True
        else:
            self.baseline = self.baseline_decay * self.baseline + (1 - self.baseline_decay) * reward
        
        advantage = reward - self.baseline
        
        # Policy gradient
        loss = -advantage * log_prob
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        return loss.item()


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("Testing PolicyNetwork and BatchReinforceTrainer")
    print("=" * 60)
    
    # Create network
    policy = PolicyNetwork()
    print(f"\nNetwork: {sum(p.numel() for p in policy.parameters()):,} parameters")
    
    # Create batch trainer
    trainer = BatchReinforceTrainer(
        policy, 
        batch_size=5,
        num_updates_per_batch=2,
    )
    
    # Simulate episodes
    print("\nSimulating episodes...")
    for ep in range(15):
        features = torch.randn(1, NUM_FLOWS * NUM_FEATURES)
        scores = policy(features)
        selected, log_prob = policy.sample_action(scores, k=12)
        
        # Fake reward
        reward = -0.03 + 0.01 * np.random.randn()
        
        trainer.store_episode(features, selected, log_prob, reward, 1.0)
        
        if trainer.should_update():
            metrics = trainer.update()
            print(f"  Batch update {metrics['num_updates']}: "
                  f"loss={metrics['loss']:.4f}, "
                  f"reward_mean={metrics['batch_reward_mean']:.4f}")
    
    print("\nDone!")