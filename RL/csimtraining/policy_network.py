"""
policy_network.py - Neural network for flow selection policy.

Updated for Fat-Tree topology (16 hosts, 240 flows).

The policy network scores all 240 flows based on their features,
then samples K=12 flows to optimize. Uses REINFORCE for training.

Architecture:
    Input: 240 flows × 9 features = 2160 dimensions (flattened)
    Hidden: 512 -> 256 -> 128 (with ReLU)
    Output: 240 scores (one per flow)
    
    Scores are converted to probabilities via softmax with temperature,
    then K flows are sampled without replacement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Tuple, List, Optional
from pathlib import Path


# =============================================================================
# Constants (Updated for Fat-Tree k=4)
# =============================================================================

NUM_HOSTS = 16
NUM_FLOWS = NUM_HOSTS * (NUM_HOSTS - 1)  # 240
NUM_FEATURES = 9
K_CRITICAL = 12  # Number of flows to select (12 of 60 packets = 20%)


# =============================================================================
# Policy Network
# =============================================================================

class PolicyNetwork(nn.Module):
    """
    Policy network for CFR-RL flow selection.
    
    Takes features for all flows and outputs scores used for sampling
    which K flows to optimize.
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
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        # Output layer (one score per flow)
        layers.append(nn.Linear(prev_dim, num_flows))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: Tensor of shape (batch, num_flows, num_features)
                      or (num_flows, num_features) for single sample
        
        Returns:
            scores: Tensor of shape (batch, num_flows) or (num_flows,)
        """
        # Handle single sample vs batch
        single_sample = features.dim() == 2
        if single_sample:
            features = features.unsqueeze(0)
        
        batch_size = features.shape[0]
        
        # Flatten: (batch, num_flows, num_features) -> (batch, num_flows * num_features)
        x = features.view(batch_size, -1)
        
        # Forward through network
        scores = self.network(x)
        
        if single_sample:
            scores = scores.squeeze(0)
        
        return scores
    
    def get_action(
        self,
        features: torch.Tensor,
        k: int = K_CRITICAL,
        temperature: float = 1.0,
        deterministic: bool = False,
    ) -> Tuple[List[int], torch.Tensor]:
        """
        Sample K flows based on scores.
        
        Args:
            features: Flow features (num_flows, num_features)
            k: Number of flows to select
            temperature: Softmax temperature (lower = more greedy)
            deterministic: If True, select top-k instead of sampling
        
        Returns:
            selected_flows: List of K flow IDs
            log_prob: Log probability of this selection (for REINFORCE)
        """
        scores = self.forward(features)
        
        if deterministic:
            # Greedy: select top-k
            _, indices = torch.topk(scores, k)
            selected_flows = indices.tolist()
            log_prob = torch.tensor(0.0)  # No gradient for deterministic
        else:
            # Sample without replacement using Gumbel-top-k trick
            selected_flows, log_prob = self._sample_k_flows(scores, k, temperature)
        
        return selected_flows, log_prob
    
    def sample_action(
        self,
        logits: torch.Tensor,
        k: int = K_CRITICAL,
        temperature: float = 1.0,
    ) -> Tuple[List[int], torch.Tensor]:
        """
        Sample K flows from pre-computed logits.
        
        Alias for _sample_k_flows with different interface.
        """
        # Handle batch dimension
        if logits.dim() == 2:
            logits = logits.squeeze(0)
        return self._sample_k_flows(logits, k, temperature)
    
    def _sample_k_flows(
        self,
        scores: torch.Tensor,
        k: int,
        temperature: float,
    ) -> Tuple[List[int], torch.Tensor]:
        """
        Sample K flows without replacement.
        
        Uses sequential sampling with masking.
        """
        device = scores.device
        num_flows = scores.shape[0]
        
        # Apply temperature
        logits = scores / temperature
        
        selected = []
        log_probs = []
        mask = torch.zeros(num_flows, device=device)
        
        for _ in range(k):
            # Mask already selected flows
            masked_logits = logits - mask * 1e9
            
            # Sample from categorical
            probs = F.softmax(masked_logits, dim=0)
            dist = Categorical(probs)
            
            idx = dist.sample()
            log_prob = dist.log_prob(idx)
            
            selected.append(idx.item())
            log_probs.append(log_prob)
            
            # Update mask
            mask[idx] = 1.0
        
        # Total log probability is sum of individual log probs
        total_log_prob = torch.stack(log_probs).sum()
        
        return selected, total_log_prob
    
    def evaluate_action(
        self,
        features: torch.Tensor,
        selected_flows: List[int],
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute log probability of a given action (for importance sampling).
        
        Args:
            features: Flow features
            selected_flows: List of K flow IDs that were selected
            temperature: Temperature used during selection
        
        Returns:
            log_prob: Log probability of this selection
        """
        scores = self.forward(features)
        logits = scores / temperature
        
        log_probs = []
        mask = torch.zeros(self.num_flows, device=scores.device)
        
        for flow_id in selected_flows:
            masked_logits = logits - mask * 1e9
            probs = F.softmax(masked_logits, dim=0)
            
            log_prob = torch.log(probs[flow_id] + 1e-10)
            log_probs.append(log_prob)
            
            mask[flow_id] = 1.0
        
        return torch.stack(log_probs).sum()


# =============================================================================
# Flow-wise Policy Network (Alternative Architecture)
# =============================================================================

class FlowWisePolicyNetwork(nn.Module):
    """
    Alternative architecture: shared encoder per flow.
    
    More parameter-efficient and better generalization,
    but may be slower due to sequential processing.
    
    Architecture:
        Per-flow encoder (shared weights):
            9 features -> 64 -> 32 -> 1 score
        Apply to all 240 flows -> 240 scores
    """
    
    def __init__(
        self,
        num_flows: int = NUM_FLOWS,
        num_features: int = NUM_FEATURES,
        hidden_dims: List[int] = [64, 32],
    ):
        super().__init__()
        
        self.num_flows = num_flows
        self.num_features = num_features
        
        # Shared encoder for each flow
        layers = []
        prev_dim = num_features
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (num_flows, num_features) or (batch, num_flows, num_features)
        
        Returns:
            scores: (num_flows,) or (batch, num_flows)
        """
        single_sample = features.dim() == 2
        if single_sample:
            features = features.unsqueeze(0)
        
        # Apply encoder to each flow
        # features: (batch, num_flows, num_features)
        scores = self.encoder(features).squeeze(-1)  # (batch, num_flows)
        
        if single_sample:
            scores = scores.squeeze(0)
        
        return scores
    
    # Inherit get_action and other methods from PolicyNetwork
    get_action = PolicyNetwork.get_action
    sample_action = PolicyNetwork.sample_action
    _sample_k_flows = PolicyNetwork._sample_k_flows
    evaluate_action = PolicyNetwork.evaluate_action


# =============================================================================
# REINFORCE Trainer
# =============================================================================

class ReinforceTrainer:
    """
    REINFORCE trainer for the policy network.
    
    Uses baseline subtraction for variance reduction.
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
        
        # Running baseline for variance reduction
        self.baseline = 0.0
        self.baseline_initialized = False
    
    def update(
        self,
        reward: float,
        log_prob: torch.Tensor,
    ) -> float:
        """
        Simplified REINFORCE update using pre-computed log_prob.
        
        Args:
            reward: Reward received
            log_prob: Log probability of the action (from policy)
        
        Returns:
            Loss value
        """
        # Update baseline
        if not self.baseline_initialized:
            self.baseline = reward
            self.baseline_initialized = True
        else:
            self.baseline = self.baseline_decay * self.baseline + (1 - self.baseline_decay) * reward
        
        # Compute advantage
        advantage = reward - self.baseline
        
        # Policy gradient loss
        loss = -advantage * log_prob
        
        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
        
        return loss.item()
    
    def update_full(
        self,
        features: torch.Tensor,
        selected_flows: List[int],
        reward: float,
        temperature: float = 1.0,
    ) -> dict:
        """
        Full REINFORCE update with entropy bonus.
        
        Args:
            features: Flow features used for action
            selected_flows: Flows that were selected
            reward: Reward received
            temperature: Temperature used during selection
        
        Returns:
            Dictionary with loss and other metrics
        """
        # Update baseline
        if not self.baseline_initialized:
            self.baseline = reward
            self.baseline_initialized = True
        else:
            self.baseline = self.baseline_decay * self.baseline + (1 - self.baseline_decay) * reward
        
        # Compute advantage
        advantage = reward - self.baseline
        
        # Compute log probability of action
        log_prob = self.policy.evaluate_action(features, selected_flows, temperature)
        
        # Policy gradient loss
        policy_loss = -advantage * log_prob
        
        # Entropy bonus (encourage exploration)
        scores = self.policy(features)
        probs = F.softmax(scores / temperature, dim=0)
        entropy = -(probs * torch.log(probs + 1e-10)).sum()
        entropy_loss = -self.entropy_weight * entropy
        
        # Total loss
        loss = policy_loss + entropy_loss
        
        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.max_grad_norm
        )
        
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'entropy': entropy.item(),
            'advantage': advantage,
            'baseline': self.baseline,
            'grad_norm': grad_norm.item(),
            'log_prob': log_prob.item(),
        }
    
    def save(self, filepath: str) -> None:
        """Save policy and optimizer state."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'baseline': self.baseline,
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """Load policy and optimizer state."""
        checkpoint = torch.load(filepath)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.baseline = checkpoint.get('baseline', 0.0)
        self.baseline_initialized = True


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("Testing PolicyNetwork (Fat-Tree)")
    print("=" * 50)
    
    # Create network
    policy = PolicyNetwork()
    print(f"\nNetwork architecture:")
    print(f"  Topology: Fat-Tree k=4 ({NUM_HOSTS} hosts)")
    print(f"  Input: {policy.input_dim} ({NUM_FLOWS} flows × {NUM_FEATURES} features)")
    print(f"  Parameters: {sum(p.numel() for p in policy.parameters()):,}")
    
    # Test forward pass
    features = torch.randn(NUM_FLOWS, NUM_FEATURES)
    scores = policy(features)
    print(f"\nForward pass:")
    print(f"  Input shape: {features.shape}")
    print(f"  Output shape: {scores.shape}")
    print(f"  Score range: [{scores.min():.3f}, {scores.max():.3f}]")
    
    # Test action sampling
    selected, log_prob = policy.get_action(features, k=K_CRITICAL, temperature=1.0)
    print(f"\nAction sampling (k={K_CRITICAL}):")
    print(f"  Selected flows: {selected}")
    print(f"  Log prob: {log_prob.item():.4f}")
    
    # Test deterministic
    selected_det, _ = policy.get_action(features, k=K_CRITICAL, deterministic=True)
    print(f"  Deterministic top-{K_CRITICAL}: {selected_det}")
    
    # Test trainer
    print("\nTesting REINFORCE update:")
    trainer = ReinforceTrainer(policy, lr=1e-4)
    
    for i in range(5):
        selected, log_prob = policy.get_action(features, k=K_CRITICAL, temperature=1.0)
        reward = -np.random.uniform(0, 0.1)  # Fake reward
        
        loss = trainer.update(reward, log_prob)
        print(f"  Step {i+1}: loss={loss:.4f}, baseline={trainer.baseline:.4f}")
    
    # Test flow-wise network
    print("\n" + "=" * 50)
    print("Testing FlowWisePolicyNetwork")
    
    flow_policy = FlowWisePolicyNetwork()
    print(f"\nParameters: {sum(p.numel() for p in flow_policy.parameters()):,}")
    
    scores = flow_policy(features)
    print(f"Output shape: {scores.shape}")
    
    selected, log_prob = flow_policy.get_action(features, k=K_CRITICAL)
    print(f"Selected flows: {selected}")