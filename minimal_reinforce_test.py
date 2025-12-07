"""
MINIMAL REINFORCE TEST

This is a stripped-down REINFORCE implementation to verify the core algorithm works.
If this works but the full implementation doesn't, we know where to look for bugs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Simple settings
NUM_FLOWS = 50      # Smaller for faster testing
NUM_FEATURES = 5
K = 5               # Select 5 flows
LR = 0.01           # Higher learning rate
NUM_EPISODES = 300

print("=" * 60)
print("MINIMAL REINFORCE TEST")
print("=" * 60)

# Very simple policy: just a linear layer
class SimplePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(NUM_FLOWS * NUM_FEATURES, NUM_FLOWS)
        # Initialize with small weights
        nn.init.normal_(self.fc.weight, mean=0, std=0.1)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, x):
        return self.fc(x)

policy = SimplePolicy()
optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

# Synthetic reward: reward = mean feature sum of selected flows
# Optimal policy: select flows with highest feature sums
def get_reward(features, selected_idx):
    """Higher reward for selecting flows with higher feature sums."""
    feat_2d = features.view(NUM_FLOWS, NUM_FEATURES)
    flow_scores = feat_2d.sum(dim=1)  # Sum features per flow
    return flow_scores[selected_idx].mean().item()

print(f"\nSetup: {NUM_FLOWS} flows, select {K}, {NUM_EPISODES} episodes")
print(f"Optimal strategy: select flows with highest feature sums")
print()

# Track rewards
all_rewards = []
all_losses = []

for ep in range(NUM_EPISODES):
    # Random features each episode
    features = torch.randn(1, NUM_FLOWS * NUM_FEATURES)
    
    # Forward pass (WITH gradient)
    scores = policy(features).squeeze()
    
    # Sample action using Gumbel-softmax
    gumbel = -torch.log(-torch.log(torch.rand_like(scores) + 1e-10) + 1e-10)
    noisy_scores = scores + gumbel
    _, selected_idx = torch.topk(noisy_scores, K)
    
    # Compute log probability
    log_probs = F.log_softmax(scores, dim=0)
    log_prob = log_probs[selected_idx].sum()
    
    # Get reward
    reward = get_reward(features, selected_idx)
    all_rewards.append(reward)
    
    # Simple baseline: running mean of rewards
    baseline = np.mean(all_rewards[-50:]) if len(all_rewards) > 1 else 0
    advantage = reward - baseline
    
    # REINFORCE loss: -advantage * log_prob
    loss = -advantage * log_prob
    
    # Gradient step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    all_losses.append(loss.item())
    
    # Progress
    if (ep + 1) % 30 == 0:
        recent_reward = np.mean(all_rewards[-30:])
        recent_loss = np.mean(all_losses[-30:])
        print(f"Episode {ep+1:3d}: reward={reward:.3f}, "
              f"recent_avg={recent_reward:.3f}, loss={recent_loss:.3f}")

# Final analysis
print("\n" + "=" * 60)
first_30 = np.mean(all_rewards[:30])
last_30 = np.mean(all_rewards[-30:])
improvement = last_30 - first_30

print(f"First 30 episodes avg reward: {first_30:.4f}")
print(f"Last 30 episodes avg reward:  {last_30:.4f}")
print(f"Improvement: {improvement:+.4f}")

# Compare to random baseline
print("\nRandom baseline check:")
random_rewards = []
for _ in range(100):
    features = torch.randn(1, NUM_FLOWS * NUM_FEATURES)
    random_idx = torch.randperm(NUM_FLOWS)[:K]
    random_rewards.append(get_reward(features, random_idx))
print(f"Random selection avg reward: {np.mean(random_rewards):.4f}")

# Compare to optimal
print("\nOptimal check:")
optimal_rewards = []
for _ in range(100):
    features = torch.randn(1, NUM_FLOWS * NUM_FEATURES)
    feat_2d = features.view(NUM_FLOWS, NUM_FEATURES)
    flow_scores = feat_2d.sum(dim=1)
    _, best_idx = torch.topk(flow_scores, K)
    optimal_rewards.append(get_reward(features, best_idx))
print(f"Optimal selection avg reward: {np.mean(optimal_rewards):.4f}")

print("\n" + "=" * 60)
if last_30 > first_30 + 0.1:
    print("✓ SUCCESS: Learning is working!")
elif last_30 > np.mean(random_rewards) + 0.1:
    print("✓ PARTIAL: Better than random but slow learning")
else:
    print("✗ FAILURE: Not learning")
print("=" * 60)