"""
Policy Network for Critical Flow Selection.
Neural network that outputs probability distribution over flows.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
import numpy as np


class CriticalFlowPolicy(nn.Module):
    """
    Policy network for selecting K critical flows.
    
    Input: Normalized traffic vector (n_flows,)
    Output: Probability distribution over flows (n_flows,)
    
    Selection: Sample K flows without replacement using the probabilities.
    """
    
    def __init__(self, n_flows: int, hidden_dim: int = 128, n_layers: int = 2):
        """
        Args:
            n_flows: Number of possible flows (n_hosts * (n_hosts - 1))
            hidden_dim: Hidden layer dimension
            n_layers: Number of hidden layers
        """
        super().__init__()
        self.n_flows = n_flows
        
        layers = []
        in_dim = n_flows
        
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.1))
            in_dim = hidden_dim
            
        layers.append(nn.Linear(hidden_dim, n_flows))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Normalized traffic vector, shape (batch, n_flows) or (n_flows,)
            
        Returns:
            Log probabilities over flows, shape (batch, n_flows) or (n_flows,)
        """
        logits = self.network(x)
        return F.log_softmax(logits, dim=-1)
    
    def get_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Get probabilities (not log)."""
        return torch.exp(self.forward(x))
    
    def select_critical_flows(self, traffic_vec: np.ndarray, k: int,
                             deterministic: bool = False
                             ) -> Tuple[List[int], torch.Tensor]:
        """
        Select K critical flows from traffic vector.
        
        Args:
            traffic_vec: Normalized traffic vector (n_flows,)
            k: Number of flows to select
            deterministic: If True, select top-k by probability
            
        Returns:
            selected_indices: List of K flow indices
            log_prob: Log probability of this selection (for REINFORCE)
        """
        x = torch.FloatTensor(traffic_vec)
        
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension
            
        with torch.no_grad() if deterministic else torch.enable_grad():
            log_probs = self.forward(x).squeeze(0)  # (n_flows,)
            probs = torch.exp(log_probs)
        
        selected = []
        total_log_prob = torch.tensor(0.0)
        
        remaining_probs = probs.clone()
        
        for _ in range(k):
            # Renormalize remaining probabilities
            remaining_probs = remaining_probs / remaining_probs.sum()
            
            if deterministic:
                # Select highest probability
                idx = remaining_probs.argmax().item()
            else:
                # Sample from distribution
                idx = torch.multinomial(remaining_probs, 1).item()
            
            selected.append(idx)
            
            # Add log prob of this selection
            total_log_prob = total_log_prob + torch.log(remaining_probs[idx] + 1e-10)
            
            # Zero out selected flow
            remaining_probs[idx] = 0.0
            
        return selected, total_log_prob


class PolicyWithBaseline(nn.Module):
    """
    Policy network with value baseline for variance reduction.
    """
    
    def __init__(self, n_flows: int, hidden_dim: int = 128):
        super().__init__()
        self.n_flows = n_flows
        
        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(n_flows, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
        )
        
        # Policy head
        self.policy_head = nn.Linear(hidden_dim, n_flows)
        
        # Value head (baseline)
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            log_probs: Log probabilities over flows
            value: Baseline value estimate
        """
        features = self.shared(x)
        log_probs = F.log_softmax(self.policy_head(features), dim=-1)
        value = self.value_head(features)
        return log_probs, value

    def get_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Get probabilities (not log)."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        log_probs, _ = self.forward(x)
        return torch.exp(log_probs.squeeze(0))
    
    def select_critical_flows(self, traffic_vec: np.ndarray, k: int,
                             deterministic: bool = False
                             ) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
        """
        Select K critical flows.
        
        Returns:
            selected_indices: List of K flow indices  
            log_prob: Log probability of selection
            value: Baseline value estimate
        """
        x = torch.FloatTensor(traffic_vec)
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        log_probs, value = self.forward(x)
        log_probs = log_probs.squeeze(0)
        value = value.squeeze()
        
        probs = torch.exp(log_probs)
        selected = []
        total_log_prob = torch.tensor(0.0)
        remaining_probs = probs.clone()
        
        for _ in range(k):
            remaining_probs = remaining_probs / (remaining_probs.sum() + 1e-10)
            
            if deterministic:
                idx = remaining_probs.argmax().item()
            else:
                idx = torch.multinomial(remaining_probs, 1).item()
                
            selected.append(idx)
            total_log_prob = total_log_prob + torch.log(remaining_probs[idx] + 1e-10)
            remaining_probs[idx] = 0.0
            
        return selected, total_log_prob, value


def test_policy():
    """Test policy network."""
    n_flows = 240  # 16 hosts, 16*15 flows
    k = 5
    
    policy = CriticalFlowPolicy(n_flows, hidden_dim=128)
    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters())}")
    
    # Random traffic
    traffic = np.random.exponential(1.0, n_flows)
    traffic = traffic / traffic.sum()
    
    # Select flows
    selected, log_prob = policy.select_critical_flows(traffic, k)
    print(f"Selected flows: {selected}")
    print(f"Log probability: {log_prob.item():.4f}")
    
    # Test with baseline
    policy_bl = PolicyWithBaseline(n_flows)
    selected, log_prob, value = policy_bl.select_critical_flows(traffic, k)
    print(f"With baseline - Value: {value.item():.4f}")


if __name__ == "__main__":
    test_policy()