"""
test_training_setup.py - Verify all training components work before full training.

Run this script to check:
1. Workload generation works
2. Feature extraction works
3. Policy network works
4. CloudSim subprocess works
5. Results parsing works

Usage:
    cd /path/to/cloudsimsdn
    python RL/test_training_setup.py
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all modules can be imported."""
    print("1. Testing imports...")
    
    try:
        from workload_generator import generate_workload, generate_demand_vector
        print("   ✓ workload_generator")
    except ImportError as e:
        print(f"   ✗ workload_generator: {e}")
        return False
    
    try:
        from feature_extractor import FeatureExtractor, NUM_FLOWS
        print("   ✓ feature_extractor")
    except ImportError as e:
        print(f"   ✗ feature_extractor: {e}")
        return False
    
    try:
        from policy_network import PolicyNetwork, ReinforceTrainer
        print("   ✓ policy_network")
    except ImportError as e:
        print(f"   ✗ policy_network: {e}")
        return False
    
    try:
        from episode_runner import EpisodeRunner, compute_reward
        print("   ✓ episode_runner")
    except ImportError as e:
        print(f"   ✗ episode_runner: {e}")
        return False
    
    try:
        import torch
        import numpy as np
        import pandas as pd
        print("   ✓ torch, numpy, pandas")
    except ImportError as e:
        print(f"   ✗ dependencies: {e}")
        return False
    
    return True


def test_workload_generation():
    """Test workload generation."""
    print("\n2. Testing workload generation...")
    
    from workload_generator import generate_workload, generate_demand_vector, get_workload_stats
    
    df = generate_workload(num_packets=100, duration=30, seed=42)
    stats = get_workload_stats(df)
    
    print(f"   Generated {stats['num_packets']} packets")
    print(f"   Active flows: {stats['num_active_flows']}")
    print(f"   Total bytes: {stats['total_bytes'] / 1e9:.2f} GB")
    
    demand = generate_demand_vector(df)
    print(f"   Demand vector shape: {demand.shape}")
    print(f"   Non-zero flows: {(demand > 0).sum()}")
    
    return True


def test_feature_extraction():
    """Test feature extraction."""
    print("\n3. Testing feature extraction...")
    
    from workload_generator import generate_workload, generate_demand_vector
    from feature_extractor import FeatureExtractor, NUM_FLOWS, NUM_FEATURES
    
    extractor = FeatureExtractor(random_cold_start=True, seed=42)
    
    df = generate_workload(num_packets=100, duration=30, seed=42)
    demand = generate_demand_vector(df)
    
    features = extractor.extract_features(demand)
    
    print(f"   Features shape: {features.shape}")
    print(f"   Expected: ({NUM_FLOWS}, {NUM_FEATURES})")
    print(f"   Feature range: [{features.min():.4f}, {features.max():.4f}]")
    
    if features.shape != (NUM_FLOWS, NUM_FEATURES):
        print("   ✗ Shape mismatch!")
        return False
    
    print("   ✓ Features extracted correctly")
    return True


def test_policy_network():
    """Test policy network."""
    print("\n4. Testing policy network...")
    
    import torch
    from feature_extractor import NUM_FLOWS, NUM_FEATURES
    from policy_network import PolicyNetwork, ReinforceTrainer
    
    policy = PolicyNetwork()
    params = sum(p.numel() for p in policy.parameters())
    print(f"   Network parameters: {params:,}")
    
    # Test forward pass
    features = torch.randn(NUM_FLOWS, NUM_FEATURES)
    scores = policy(features)
    print(f"   Forward pass: {features.shape} -> {scores.shape}")
    
    # Test action sampling
    selected, log_prob = policy.get_action(features, k=8, temperature=1.0)
    print(f"   Selected flows: {selected}")
    print(f"   Log prob: {log_prob.item():.4f}")
    
    # Test training step
    trainer = ReinforceTrainer(policy, lr=1e-4)
    metrics = trainer.update(features, selected, reward=-0.01, temperature=1.0)
    print(f"   Training step loss: {metrics['loss']:.6f}")
    
    print("   ✓ Policy network works")
    return True


def test_cloudsim_runner(cloudsim_dir: str = "."):
    """Test CloudSim subprocess."""
    print("\n5. Testing CloudSim runner...")
    
    from episode_runner import EpisodeRunner
    from workload_generator import generate_workload, save_workload
    
    # Check CloudSim directory
    cloudsim_path = Path(cloudsim_dir).resolve()
    pom_file = cloudsim_path / "pom.xml"
    
    if not pom_file.exists():
        print(f"   ✗ pom.xml not found in {cloudsim_dir}")
        print("   Make sure you're running from the CloudSimSDN directory")
        return False
    
    print(f"   CloudSim directory: {cloudsim_path}")
    
    # Generate test workload
    df = generate_workload(num_packets=50, duration=30, seed=42)
    test_dir = cloudsim_path / "test_setup"
    test_dir.mkdir(exist_ok=True)
    
    # Create result directory that CloudSim will use
    result_dir = cloudsim_path / "result_test_setup"
    result_dir.mkdir(exist_ok=True)
    
    workload_file = test_dir / "test_workload.csv"
    save_workload(df, str(workload_file))
    print(f"   Created test workload: {workload_file}")
    print(f"   First 3 rows of workload:")
    print(df.head(3).to_string(index=False))
    
    # Write critical flows
    critical_flows = [84, 50, 109, 17, 23, 91, 42, 7]
    critical_file = test_dir / "critical_flows.txt"
    with open(critical_file, 'w') as f:
        for flow_id in critical_flows:
            f.write(f"{flow_id}\n")
    print(f"   Created critical flows: {critical_file}")
    
    # Run episode
    print("   Running CloudSim episode (this may take a few seconds)...")
    runner = EpisodeRunner(cloudsim_dir=str(cloudsim_path), verbose=False)
    
    # Use path relative to cloudsim_dir
    results = runner.run_episode(
        workload_file="test_setup/test_workload.csv",
        critical_flows=critical_flows,
        output_dir="test_setup/output",
        episode_id=0,
    )
    
    if not results['success']:
        print(f"   ✗ CloudSim failed: {results.get('error', 'Unknown')}")
        if 'stderr' in results:
            print(f"   stderr: {results['stderr'][:500]}")
        return False
    
    print(f"   ✓ Episode completed in {results['wall_time_ms']}ms")
    print(f"   Mean queuing: {results['episode_summary'].get('mean_queuing_ms', 0):.2f}ms")
    print(f"   Total packets: {results['episode_summary'].get('total_packets', 0)}")
    
    # Test reward computation
    from episode_runner import compute_reward
    reward = compute_reward(results['episode_summary'])
    print(f"   Computed reward: {reward:.6f}")
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir)
    if result_dir.exists():
        shutil.rmtree(result_dir)
    print("   Cleaned up test files")
    
    return True


def test_full_pipeline(cloudsim_dir: str = "."):
    """Test the full training pipeline with one episode."""
    print("\n6. Testing full pipeline (1 episode)...")
    
    import torch
    from workload_generator import generate_workload, save_workload, generate_demand_vector
    from feature_extractor import FeatureExtractor
    from policy_network import PolicyNetwork, ReinforceTrainer
    from episode_runner import EpisodeRunner, compute_reward
    
    # Initialize components
    extractor = FeatureExtractor(random_cold_start=True)
    policy = PolicyNetwork()
    trainer = ReinforceTrainer(policy, lr=1e-4)
    runner = EpisodeRunner(cloudsim_dir=cloudsim_dir, verbose=False)
    
    cloudsim_path = Path(cloudsim_dir)
    test_dir = cloudsim_path / "test_pipeline"
    test_dir.mkdir(exist_ok=True)
    
    # 1. Generate workload
    df = generate_workload(num_packets=50, duration=30)
    workload_file = test_dir / "workload.csv"
    save_workload(df, str(workload_file))
    
    # 2. Extract features
    demand = generate_demand_vector(df)
    features = extractor.extract_features(demand)
    features_tensor = torch.tensor(features, dtype=torch.float32)
    
    # 3. Select flows
    selected_flows, log_prob = policy.get_action(features_tensor, k=8, temperature=1.0)
    print(f"   Selected flows: {selected_flows}")
    
    # 4. Run CloudSim
    print("   Running CloudSim...")
    results = runner.run_episode(
        workload_file=str(workload_file.relative_to(cloudsim_path)),
        critical_flows=selected_flows,
        output_dir="test_pipeline/output",
    )
    
    if not results['success']:
        print(f"   ✗ CloudSim failed: {results.get('error')}")
        return False
    
    # 5. Compute reward
    reward = compute_reward(results['episode_summary'])
    print(f"   Reward: {reward:.6f}")
    
    # 6. Update policy
    metrics = trainer.update(features_tensor, selected_flows, reward, temperature=1.0)
    print(f"   Loss: {metrics['loss']:.6f}")
    print(f"   Baseline: {metrics['baseline']:.6f}")
    
    # 7. Update history
    extractor.update_history(
        results['flow_summary'],
        results['link_stats'],
        results['episode_summary']
    )
    print("   ✓ History updated")
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir)
    
    print("   ✓ Full pipeline works!")
    return True


def main():
    print("=" * 60)
    print("CloudSim Training Setup Test")
    print("=" * 60)
    
    # Get CloudSim directory
    cloudsim_dir = "."
    if len(sys.argv) > 1:
        cloudsim_dir = sys.argv[1]
    
    all_passed = True
    
    # Run tests
    if not test_imports():
        all_passed = False
        print("\n✗ Import test failed. Install missing dependencies.")
        return
    
    if not test_workload_generation():
        all_passed = False
    
    if not test_feature_extraction():
        all_passed = False
    
    if not test_policy_network():
        all_passed = False
    
    if not test_cloudsim_runner(cloudsim_dir):
        all_passed = False
        print("\n✗ CloudSim test failed. Check Java setup.")
    else:
        # Only run full pipeline if CloudSim works
        if not test_full_pipeline(cloudsim_dir):
            all_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("\nYou can now run full training:")
        print(f"  python RL/cloudsim_trainer.py --cloudsim-dir {cloudsim_dir} --episodes 500")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nFix the issues above before running full training.")
    print("=" * 60)


if __name__ == "__main__":
    main()