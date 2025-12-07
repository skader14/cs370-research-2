"""
DEFINITIVE DIAGNOSTIC TEST FOR CFR-RL LEARNING

This script tests THREE hypotheses:

HYPOTHESIS 1: Policy actions have no effect on CloudSim outcomes
TEST: Run same workload with different critical flow selections
EXPECTED: If policy matters, different selections → different rewards
ACTUAL: If rewards are same, policy has no effect

HYPOTHESIS 2: Reward variance comes from workload randomness, not policy
TEST: Run same workload multiple times with same critical flows  
EXPECTED: If deterministic, same workload → same reward
ACTUAL: Confirms/denies CloudSim determinism

HYPOTHESIS 3: Training loop itself is broken
TEST: Use synthetic reward function instead of CloudSim
EXPECTED: If training works, policy should learn synthetic function
ACTUAL: Isolates Python code from CloudSim issues

Run this BEFORE more training experiments.
"""

import numpy as np
import torch
import sys
import os

# Add training code path
sys.path.insert(0, 'RL/csimtraining')

def test_hypothesis_1():
    """
    Test if different critical flow selections produce different rewards.
    
    This requires running CloudSim twice with:
    - Same workload file
    - Different critical_flows.txt
    
    Manual test - provides instructions.
    """
    print("=" * 70)
    print("HYPOTHESIS 1: Do policy actions affect CloudSim outcomes?")
    print("=" * 70)
    print("""
MANUAL TEST PROCEDURE:

1. Generate a workload file (or use one from a previous episode):
   cp training_outputs/episodes/ep_0000/workload.csv test_workload.csv

2. Create two different critical flow selections:
   
   critical_A.txt (random selection):
   1
   5
   10
   15
   20
   25
   30
   35
   40
   45
   50
   55
   
   critical_B.txt (different selection):
   100
   105
   110
   115
   120
   125
   130
   135
   140
   145
   150
   155

3. Run CloudSim twice with same workload but different critical flows:
   
   java -cp "target/classes;..." org.cloudbus.cloudsim.sdn.example.CFRRLTrainingRunner \\
       test_workload.csv critical_A.txt outputs_A/
   
   java -cp "target/classes;..." org.cloudbus.cloudsim.sdn.example.CFRRLTrainingRunner \\
       test_workload.csv critical_B.txt outputs_B/

4. Compare episode_summary.json from both runs:
   - If mean_queuing_delay is IDENTICAL: Policy has NO effect
   - If mean_queuing_delay is DIFFERENT: Policy affects outcomes

5. Check routing statistics in episode.log:
   - Critical flow routings should be different
   - If total outcomes are same, routing differences don't matter
""")
    return None


def test_hypothesis_2():
    """
    Test if CloudSim is deterministic.
    
    Run same workload + same critical flows twice.
    If results differ, there's randomness in CloudSim.
    """
    print("=" * 70)
    print("HYPOTHESIS 2: Is CloudSim deterministic?")
    print("=" * 70)
    print("""
MANUAL TEST PROCEDURE:

1. Use the SAME workload.csv and critical_flows.txt

2. Run CloudSim twice:
   
   java -cp ... CFRRLTrainingRunner workload.csv critical.txt outputs_run1/
   java -cp ... CFRRLTrainingRunner workload.csv critical.txt outputs_run2/

3. Compare results:
   - If IDENTICAL: CloudSim is deterministic (good)
   - If DIFFERENT: CloudSim has internal randomness (complicates learning)
""")
    return None


def test_hypothesis_3():
    """
    Test if the training loop can learn ANYTHING.
    
    Replace CloudSim reward with a synthetic function the policy
    SHOULD be able to learn.
    """
    print("=" * 70)
    print("HYPOTHESIS 3: Can the training loop learn at all?")
    print("=" * 70)
    
    try:
        from policy_network_fixed import PolicyNetwork, BatchReinforceTrainer
        print("Using FIXED policy network")
    except ImportError:
        try:
            from policy_network import PolicyNetwork, BatchReinforceTrainer
            print("Using original policy network")
        except ImportError:
            print("ERROR: Could not import policy_network. Run from CloudSimSDN directory.")
            return False
    
    print("\nRunning synthetic reward test...")
    print("If this works, the Python training code is fine.\n")
    
    # Create policy and trainer
    NUM_FLOWS = 240
    NUM_FEATURES = 9
    K_CRITICAL = 12
    
    policy = PolicyNetwork(num_flows=NUM_FLOWS, num_features=NUM_FEATURES)
    
    # Print initial weight stats to verify initialization
    total_params = sum(p.numel() for p in policy.parameters())
    weight_std = np.mean([p.std().item() for p in policy.parameters()])
    print(f"Policy network: {total_params:,} parameters")
    print(f"Average weight std: {weight_std:.4f} (should be ~0.1-0.5)")
    print()
    
    # Use higher LR and simpler settings for testing
    trainer = BatchReinforceTrainer(
        policy,
        lr=1e-3,
        batch_size=10,
        num_updates_per_batch=1,  # Simpler
        entropy_weight=0.01,
        normalize_advantages=True,
        lr_schedule=False,  # No scheduling
    )
    
    # Synthetic reward function:
    # Reward is better when policy selects flows with HIGH feature values
    def synthetic_reward(features, selected_flows):
        """
        Reward = mean feature value of selected flows.
        Optimal policy: select flows with highest feature sums.
        Random policy: expected reward ≈ 0
        Optimal policy: expected reward ≈ 0.3-0.5
        """
        features_2d = features.view(NUM_FLOWS, NUM_FEATURES)
        flow_scores = features_2d.sum(dim=1)
        selected_scores = flow_scores[selected_flows]
        return selected_scores.mean().item() / 10.0
    
    # Training loop
    rewards_history = []
    
    for episode in range(200):
        features = torch.randn(1, NUM_FLOWS * NUM_FEATURES)
        
        # Forward pass (WITH gradient for proper log_prob)
        scores = policy(features)
        selected_flows, log_prob = policy.sample_action(scores, k=K_CRITICAL, temperature=1.0)
        
        # Synthetic reward
        reward = synthetic_reward(features, selected_flows)
        rewards_history.append(reward)
        
        # Store for batch update
        trainer.store_episode(features, selected_flows, log_prob, reward, 1.0)
        
        # Batch update
        if trainer.should_update():
            metrics = trainer.update()
            
            recent_mean = np.mean(rewards_history[-20:])
            print(f"Episode {episode:3d}: reward={reward:.4f}, "
                  f"recent_avg={recent_mean:.4f}, loss={metrics['loss']:.4f}, "
                  f"entropy={metrics.get('entropy', 0):.2f}")
    
    # Analyze results
    print("\n" + "=" * 50)
    first_20 = np.mean(rewards_history[:20])
    last_20 = np.mean(rewards_history[-20:])
    improvement = last_20 - first_20
    
    print(f"First 20 episodes avg reward: {first_20:.4f}")
    print(f"Last 20 episodes avg reward:  {last_20:.4f}")
    print(f"Improvement: {improvement:+.4f}")
    
    # Also check if last 20 is significantly above 0 (random baseline)
    if last_20 > 0.05:
        print(f"\n✓ Policy is selecting better-than-random flows!")
    
    if improvement > 0.02:
        print("\n✓ SUCCESS: Training loop CAN learn!")
        print("  The Python code is working correctly.")
        print("  Problem is with CloudSim not providing learnable signal.")
        return True
    else:
        print("\n✗ FAILURE: Training loop did NOT learn")
        print("  There may be a bug in the training code.")
        return False


def main():
    print("=" * 70)
    print("CFR-RL DIAGNOSTIC TEST SUITE")
    print("=" * 70)
    print()
    
    # Test 1 and 2 are manual
    test_hypothesis_1()
    print()
    test_hypothesis_2()
    print()
    
    # Test 3 is automated
    result = test_hypothesis_3()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
If Hypothesis 3 passed (training CAN learn):
  → Problem is CloudSim not providing learning signal
  → Need to create actual congestion in simulation
  → Or verify policy actions affect CloudSim routing

If Hypothesis 3 failed (training CANNOT learn):
  → Bug in Python training code
  → Fix before continuing with CloudSim

For Hypotheses 1 & 2:
  → Run manual tests as described above
  → Results will confirm/deny policy effectiveness
""")


if __name__ == "__main__":
    main()