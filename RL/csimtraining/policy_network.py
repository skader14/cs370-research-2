"""
policy_network.py - Policy and Value Networks with PPO Trainer for CFR-RL.

This module implements:
1. PolicyNetwork (Actor) - Outputs flow selection probabilities
2. ValueNetwork (Critic) - Estimates state value for advantage computation
3. PPOTrainer - Proximal Policy Optimization with clipped objective

PPO advantages over REINFORCE:
- Lower variance through learned value baseline
- Multiple gradient updates per batch (sample efficient)
- Clipped objective prevents destructive large updates
- Generalized Advantage Estimation (GAE) for bias-variance tradeoff
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import List, Tuple, Dict, Optional
import numpy as np


# =============================================================================
# Policy Network (Actor)
# =============================================================================

class PolicyNetwork(nn.Module):
    """
    Neural network that outputs flow selection probabilities.
    
    Architecture:
        Input: [batch, num_flows * num_features] (flattened) or [batch, num_flows, num_features]
        Process: Reshape to [batch * num_flows, num_features], apply shared MLP
        Output: [batch, num_flows] - Logits for each flow
    
    The network learns to score each flow based on its features,
    then we sample K flows based on these scores.
    """
    
    def __init__(
        self,
        num_flows: int,
        num_features: int,
        hidden_dim: int = 256,
        num_hidden_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_flows = num_flows
        self.num_features = num_features
        
        # Build MLP layers
        layers = []
        input_dim = num_features
        current_hidden = hidden_dim
        
        for i in range(num_hidden_layers):
            layers.append(nn.Linear(input_dim, current_hidden))
            layers.append(nn.LayerNorm(current_hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = current_hidden
            current_hidden = current_hidden // 2 if current_hidden > 64 else current_hidden
        
        # Output layer: single logit per flow
        layers.append(nn.Linear(input_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute flow logits.
        
        Args:
            x: Flow features - either:
               - [batch, num_flows * num_features] (flattened from feature_extractor)
               - [batch, num_flows, num_features] (3D)
        
        Returns:
            logits: [batch, num_flows] - Unnormalized log probabilities
        """
        batch_size = x.shape[0]
        
        # Handle both 2D (flattened) and 3D input
        if x.dim() == 2:
            # Input is [batch, num_flows * num_features] - reshape to process per-flow
            # This reshapes [1, 2160] to [240, 9] for batch_size=1
            x_flat = x.view(-1, self.num_features)
        else:
            # Input is [batch, num_flows, num_features]
            x_flat = x.view(-1, self.num_features)
        
        # Apply MLP to each flow's features
        logits_flat = self.mlp(x_flat)  # [batch * num_flows, 1]
        
        # Reshape back to [batch, num_flows]
        logits = logits_flat.view(batch_size, self.num_flows)
        
        return logits
    
    def sample_action(
        self,
        logits: torch.Tensor,
        k: int,
        temperature: float = 1.0,
    ) -> Tuple[List[int], torch.Tensor]:
        """
        Sample K flows from the policy distribution.
        
        Uses temperature-scaled softmax sampling without replacement.
        
        Args:
            logits: [batch, num_flows] - Raw logits from forward pass
            k: Number of flows to select
            temperature: Sampling temperature (lower = more deterministic)
        
        Returns:
            selected_flows: List of K flow indices
            log_prob: Sum of log probabilities for selected flows
        """
        # Apply temperature scaling
        scaled_logits = logits / max(temperature, 1e-8)
        
        # Convert to probabilities
        probs = F.softmax(scaled_logits, dim=-1)
        
        # Sample without replacement
        selected_flows = []
        log_probs = []
        
        remaining_probs = probs.clone().squeeze(0)  # [num_flows]
        
        for _ in range(k):
            # Renormalize remaining probabilities
            remaining_probs = remaining_probs / (remaining_probs.sum() + 1e-8)
            
            # Sample one flow
            dist = Categorical(remaining_probs)
            flow_idx = dist.sample()
            
            # Record selection
            selected_flows.append(flow_idx.item())
            log_probs.append(dist.log_prob(flow_idx))
            
            # Zero out selected flow for next iteration
            remaining_probs[flow_idx] = 0
        
        # Sum log probs for joint probability
        total_log_prob = torch.stack(log_probs).sum()
        
        return selected_flows, total_log_prob
    
    def evaluate_actions(
        self,
        logits: torch.Tensor,
        actions: List[List[int]],
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities and entropy for given actions.
        
        Used during PPO update to compute importance sampling ratio.
        
        Args:
            logits: [batch, num_flows] - Raw logits
            actions: List of lists, each containing K selected flow indices
            temperature: Temperature used during action selection
        
        Returns:
            log_probs: [batch] - Log probability of each action set
            entropy: [batch] - Entropy of the distribution
        """
        batch_size = logits.shape[0]
        scaled_logits = logits / max(temperature, 1e-8)
        
        all_log_probs = []
        all_entropies = []
        
        for b in range(batch_size):
            probs = F.softmax(scaled_logits[b], dim=-1)
            
            # Compute log prob for this action sequence
            log_prob_sum = 0.0
            remaining_probs = probs.clone()
            
            for flow_idx in actions[b]:
                remaining_probs = remaining_probs / (remaining_probs.sum() + 1e-8)
                log_prob_sum = log_prob_sum + torch.log(remaining_probs[flow_idx] + 1e-8)
                remaining_probs[flow_idx] = 0
            
            all_log_probs.append(log_prob_sum)
            
            # Entropy of full distribution (before sampling)
            entropy = Categorical(probs).entropy()
            all_entropies.append(entropy)
        
        return torch.stack(all_log_probs), torch.stack(all_entropies)


# =============================================================================
# Value Network (Critic)
# =============================================================================

class ValueNetwork(nn.Module):
    """
    Neural network that estimates state value V(s).
    
    The value network predicts the expected reward from a given state,
    which is used to compute advantages for policy gradient updates.
    
    Architecture:
        Input: [batch, num_flows * num_features] (flattened) or [batch, num_flows, num_features]
        Process: Reshape to 3D, aggregate flow features (mean/std/max), then MLP
        Output: [batch, 1] - Estimated value
    """
    
    def __init__(
        self,
        num_flows: int,
        num_features: int,
        hidden_dim: int = 256,
        num_hidden_layers: int = 2,
    ):
        super().__init__()
        
        self.num_flows = num_flows
        self.num_features = num_features
        
        # Flow feature aggregation: mean + std + max pooling
        # This gives us a fixed-size representation regardless of num_flows
        aggregated_dim = num_features * 3  # mean, std, max = 9 * 3 = 27
        
        # MLP for value estimation
        layers = []
        input_dim = aggregated_dim
        current_hidden = hidden_dim
        
        for i in range(num_hidden_layers):
            layers.append(nn.Linear(input_dim, current_hidden))
            layers.append(nn.LayerNorm(current_hidden))
            layers.append(nn.ReLU())
            input_dim = current_hidden
            current_hidden = current_hidden // 2 if current_hidden > 64 else current_hidden
        
        # Output single value
        layers.append(nn.Linear(input_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Estimate value of state.
        
        Args:
            x: Flow features - either:
               - [batch, num_flows * num_features] (flattened from feature_extractor)
               - [batch, num_flows, num_features] (3D)
        
        Returns:
            value: [batch, 1]
        """
        batch_size = x.shape[0]
        
        # Reshape to 3D: [batch, num_flows, num_features]
        if x.dim() == 2:
            # Input is flattened [batch, num_flows * num_features]
            # Reshape to [batch, num_flows, num_features]
            x = x.view(batch_size, self.num_flows, self.num_features)
        
        # Now x is [batch, num_flows, num_features]
        # Aggregate across flows (dim=1): mean, std, max pooling
        mean_features = x.mean(dim=1)  # [batch, num_features]
        std_features = x.std(dim=1)    # [batch, num_features]
        max_features = x.max(dim=1)[0] # [batch, num_features]
        
        # Concatenate aggregated features: [batch, num_features * 3]
        aggregated = torch.cat([mean_features, std_features, max_features], dim=-1)
        
        # MLP to estimate value
        value = self.mlp(aggregated)
        
        return value


# =============================================================================
# PPO Trainer
# =============================================================================

class PPOTrainer:
    """
    Proximal Policy Optimization trainer.
    
    PPO improves on REINFORCE by:
    1. Using a learned value function baseline (reduces variance)
    2. Multiple gradient updates per batch (sample efficient)
    3. Clipped objective to prevent destructive updates (stable)
    4. GAE for better advantage estimation (bias-variance tradeoff)
    
    Reference: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
    """
    
    def __init__(
        self,
        policy: PolicyNetwork,
        value_net: ValueNetwork,
        lr_policy: float = 3e-4,
        lr_value: float = 1e-3,
        batch_size: int = 10,
        ppo_epochs: int = 4,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        gae_lambda: float = 0.95,
        gamma: float = 0.99,
        normalize_advantages: bool = True,
        target_kl: Optional[float] = 0.02,
    ):
        """
        Initialize PPO trainer.
        
        Args:
            policy: Policy network (actor)
            value_net: Value network (critic)
            lr_policy: Learning rate for policy
            lr_value: Learning rate for value network
            batch_size: Episodes per update batch
            ppo_epochs: Gradient updates per batch
            clip_epsilon: PPO clipping parameter
            value_loss_coef: Weight for value loss
            entropy_coef: Weight for entropy bonus
            max_grad_norm: Gradient clipping threshold
            gae_lambda: GAE lambda parameter
            gamma: Discount factor
            normalize_advantages: Whether to normalize advantages
            target_kl: Early stopping if KL exceeds this (None = disabled)
        """
        self.policy = policy
        self.value_net = value_net
        
        # Separate optimizers for actor and critic
        self.policy_optimizer = torch.optim.Adam(policy.parameters(), lr=lr_policy)
        self.value_optimizer = torch.optim.Adam(value_net.parameters(), lr=lr_value)
        
        # For compatibility with existing code
        self.optimizer = self.policy_optimizer
        
        self.batch_size = batch_size
        self.ppo_epochs = ppo_epochs
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.normalize_advantages = normalize_advantages
        self.target_kl = target_kl
        
        # Episode buffer
        self.buffer = {
            'features': [],      # State features
            'actions': [],       # Selected flows
            'log_probs': [],     # Log prob at time of selection
            'rewards': [],       # Episode rewards
            'temperatures': [],  # Temperature used
            'values': [],        # Value estimates
        }
        
        # Running baseline (exponential moving average) for logging
        self.baseline = 0.0
        self.baseline_decay = 0.99
        
        # Statistics
        self.update_count = 0
    
    def store_episode(
        self,
        features: torch.Tensor,
        selected_flows: List[int],
        log_prob: torch.Tensor,
        reward: float,
        temperature: float,
    ) -> None:
        """
        Store an episode's data for batch update.
        
        Args:
            features: [1, num_flows * num_features] - State features (flattened)
            selected_flows: List of selected flow indices
            log_prob: Log probability of the action
            reward: Episode reward
            temperature: Temperature used for sampling
        """
        # Get value estimate for this state
        with torch.no_grad():
            value = self.value_net(features).squeeze()
        
        self.buffer['features'].append(features.detach())
        self.buffer['actions'].append(selected_flows)
        self.buffer['log_probs'].append(log_prob.detach())
        self.buffer['rewards'].append(reward)
        self.buffer['temperatures'].append(temperature)
        self.buffer['values'].append(value)
        
        # Update running baseline
        self.baseline = self.baseline_decay * self.baseline + (1 - self.baseline_decay) * reward
    
    def should_update(self) -> bool:
        """Check if we have enough episodes for a batch update."""
        return len(self.buffer['rewards']) >= self.batch_size
    
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages using Generalized Advantage Estimation (GAE).
        
        For episodic tasks with single reward per episode, this simplifies to:
        advantage = reward - value
        
        Args:
            rewards: [batch] - Episode rewards
            values: [batch] - Value estimates
        
        Returns:
            advantages: [batch] - GAE advantages
            returns: [batch] - Target values for critic
        """
        # For single-step episodes (one reward per episode):
        # advantage = reward - value (no bootstrapping needed)
        # returns = reward (target for value function)
        
        returns = rewards.clone()
        advantages = rewards - values
        
        return advantages, returns
    
    def update(self) -> Dict[str, float]:
        """
        Perform PPO update using buffered episodes.
        
        Returns:
            Dictionary with training metrics
        """
        if len(self.buffer['rewards']) == 0:
            return {'loss': 0.0}
        
        # Convert buffer to tensors
        device = next(self.policy.parameters()).device
        
        features = torch.cat(self.buffer['features'], dim=0).to(device)  # [batch, num_flows * num_features]
        old_log_probs = torch.stack(self.buffer['log_probs']).to(device)  # [batch]
        rewards = torch.tensor(self.buffer['rewards'], dtype=torch.float32).to(device)
        values = torch.stack(self.buffer['values']).to(device)
        actions = self.buffer['actions']  # List of lists
        temperatures = self.buffer['temperatures']
        
        # Use mean temperature for evaluation (they should be similar within batch)
        mean_temp = np.mean(temperatures)
        
        # Compute advantages
        advantages, returns = self.compute_advantages(rewards, values)
        
        if self.normalize_advantages and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update loop
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        num_updates = 0
        
        for epoch in range(self.ppo_epochs):
            # Get current policy log probs and entropy
            logits = self.policy(features)
            new_log_probs, entropy = self.policy.evaluate_actions(
                logits, actions, temperature=mean_temp
            )
            
            # Importance sampling ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            new_values = self.value_net(features).squeeze()
            value_loss = F.mse_loss(new_values, returns)
            
            # Entropy bonus (encourages exploration)
            entropy_loss = -entropy.mean()
            
            # Combined loss
            loss = (
                policy_loss 
                + self.value_loss_coef * value_loss 
                + self.entropy_coef * entropy_loss
            )
            
            # Update networks
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
            
            self.policy_optimizer.step()
            self.value_optimizer.step()
            
            # Track metrics
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
            num_updates += 1
            
            # Early stopping based on KL divergence
            with torch.no_grad():
                kl = (old_log_probs - new_log_probs).mean().item()
                total_kl += abs(kl)
                
                if self.target_kl is not None and abs(kl) > self.target_kl:
                    print(f"    [PPO] Early stopping at epoch {epoch+1}/{self.ppo_epochs} (KL={kl:.4f})")
                    break
        
        # Clear buffer
        self._clear_buffer()
        self.update_count += 1
        
        return {
            'loss': total_policy_loss / num_updates,
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates,
            'kl_divergence': total_kl / num_updates,
            'batch_reward_mean': rewards.mean().item(),
            'batch_reward_std': rewards.std().item(),
            'advantages_mean': advantages.mean().item(),
            'num_epochs': num_updates,
        }
    
    def _clear_buffer(self) -> None:
        """Clear the episode buffer."""
        self.buffer = {
            'features': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'temperatures': [],
            'values': [],
        }
    
    def get_baseline(self) -> float:
        """Get current baseline estimate (for logging compatibility)."""
        return self.baseline
    
    def get_value_estimate(self, features: torch.Tensor) -> float:
        """Get value estimate for a state."""
        with torch.no_grad():
            return self.value_net(features).item()


# =============================================================================
# Legacy REINFORCE Trainer (kept for comparison)
# =============================================================================

class BatchReinforceTrainer:
    """
    Original REINFORCE trainer (kept for A/B comparison).
    
    This is the baseline algorithm that PPO improves upon.
    """
    
    def __init__(
        self,
        policy: PolicyNetwork,
        lr: float = 1e-4,
        batch_size: int = 10,
        num_updates_per_batch: int = 3,
        entropy_weight: float = 0.01,
        max_grad_norm: float = 0.5,
        normalize_advantages: bool = True,
        total_episodes: int = 500,
    ):
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        
        self.batch_size = batch_size
        self.num_updates_per_batch = num_updates_per_batch
        self.entropy_weight = entropy_weight
        self.max_grad_norm = max_grad_norm
        self.normalize_advantages = normalize_advantages
        
        # Episode buffer
        self.episode_buffer = []
        
        # Baseline (exponential moving average)
        self.baseline = 0.0
        self.baseline_decay = 0.99
    
    def store_episode(
        self,
        features: torch.Tensor,
        selected_flows: List[int],
        log_prob: torch.Tensor,
        reward: float,
        temperature: float,
    ) -> None:
        """Store episode data."""
        self.episode_buffer.append({
            'features': features.detach(),
            'selected_flows': selected_flows,
            'log_prob': log_prob,
            'reward': reward,
            'temperature': temperature,
        })
        
        # Update baseline
        self.baseline = self.baseline_decay * self.baseline + (1 - self.baseline_decay) * reward
    
    def should_update(self) -> bool:
        return len(self.episode_buffer) >= self.batch_size
    
    def update(self) -> Dict[str, float]:
        if len(self.episode_buffer) == 0:
            return {'loss': 0.0}
        
        device = next(self.policy.parameters()).device
        
        # Extract batch data
        rewards = torch.tensor(
            [ep['reward'] for ep in self.episode_buffer],
            dtype=torch.float32, device=device
        )
        log_probs = torch.stack([ep['log_prob'] for ep in self.episode_buffer])
        
        # Compute advantages
        advantages = rewards - self.baseline
        
        if self.normalize_advantages and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Multiple gradient updates
        total_loss = 0.0
        for _ in range(self.num_updates_per_batch):
            policy_loss = -(log_probs * advantages.detach()).mean()
            
            self.optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            total_loss += policy_loss.item()
        
        avg_loss = total_loss / self.num_updates_per_batch
        batch_reward = rewards.mean().item()
        
        # Clear buffer
        self.episode_buffer = []
        
        return {
            'loss': avg_loss,
            'batch_reward_mean': batch_reward,
        }
    
    def get_baseline(self) -> float:
        return self.baseline


# =============================================================================
# Utility Functions
# =============================================================================

def create_ppo_networks(
    num_flows: int,
    num_features: int,
    hidden_dim: int = 256,
    num_hidden_layers: int = 3,
    device: torch.device = None,
) -> Tuple[PolicyNetwork, ValueNetwork]:
    """
    Create policy and value networks for PPO.
    
    Args:
        num_flows: Number of flows in the network
        num_features: Features per flow
        hidden_dim: Hidden layer dimension
        num_hidden_layers: Number of hidden layers
        device: Device to place networks on
    
    Returns:
        (policy_network, value_network)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    policy = PolicyNetwork(
        num_flows=num_flows,
        num_features=num_features,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
    ).to(device)
    
    value_net = ValueNetwork(
        num_flows=num_flows,
        num_features=num_features,
        hidden_dim=hidden_dim,
        num_hidden_layers=max(1, num_hidden_layers - 1),  # Slightly smaller
    ).to(device)
    
    return policy, value_net