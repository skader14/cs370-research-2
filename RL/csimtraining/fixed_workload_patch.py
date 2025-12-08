"""
Fixed Workload Mixin for CFR-RL Training

Add this code to either:
1. cloudsim_trainer.py (in the CloudSimTrainer class)
2. episode_runner.py (in the EpisodeRunner class)

This implements workload reuse within batches to enable learning.
"""

# ============================================================================
# OPTION A: Add to CloudSimTrainer class in cloudsim_trainer.py
# ============================================================================

# Add these instance variables in __init__:
"""
# Fixed workload support
self.fixed_workload_mode = True  # Set to False to disable
self.workload_reuse_count = 0
self.workload_reuse_limit = self.batch_size  # Reuse for entire batch
self._cached_workload = None
self._cached_workload_path = None
"""

# Replace the workload generation in run_episode with:
"""
def _get_workload_for_episode(self, episode_id, episode_dir):
    '''Get workload - either generate new or reuse cached.'''
    
    if not self.fixed_workload_mode:
        # Original behavior - new workload each episode
        return self._generate_new_workload(episode_dir)
    
    # Fixed workload mode - reuse within batch
    batch_num = episode_id // self.batch_size
    episode_in_batch = episode_id % self.batch_size
    
    if episode_in_batch == 0:
        # First episode of batch - generate new workload
        print(f"[Trainer] Batch {batch_num}: Generating new fixed workload")
        self._cached_workload = self.workload_generator.generate(
            num_packets=self.packets_per_episode,
            duration=self.episode_duration
        )
        # Save to batch directory
        batch_workload_path = self.output_dir / f"batch_{batch_num:04d}_workload.csv"
        self._cached_workload.to_csv(batch_workload_path, index=False)
        self._cached_workload_path = batch_workload_path
    else:
        print(f"[Trainer] Batch {batch_num}: Reusing workload (ep {episode_in_batch + 1}/{self.batch_size})")
    
    # Copy cached workload to episode directory
    episode_workload_path = episode_dir / "workload.csv"
    self._cached_workload.to_csv(episode_workload_path, index=False)
    
    return episode_workload_path
"""


# ============================================================================
# OPTION B: Standalone function to patch existing code
# ============================================================================

class FixedWorkloadManager:
    """
    Manages fixed workloads for fair policy comparison.
    
    Usage:
        manager = FixedWorkloadManager(batch_size=10)
        
        for episode in range(num_episodes):
            workload_path = manager.get_workload(
                episode_id=episode,
                episode_dir=episode_dir,
                generator=workload_generator,
                num_packets=2000,
                duration=10.0
            )
    """
    
    def __init__(self, batch_size=10, output_dir=None):
        self.batch_size = batch_size
        self.output_dir = output_dir
        self._cached_workload = None
        self._current_batch = -1
    
    def get_workload(self, episode_id, episode_dir, generator, num_packets, duration):
        """Get workload for episode - generates new one at batch boundaries."""
        import shutil
        from pathlib import Path
        
        episode_dir = Path(episode_dir)
        batch_num = episode_id // self.batch_size
        
        # Check if we need a new workload (new batch)
        if batch_num != self._current_batch:
            self._current_batch = batch_num
            print(f"\n[FixedWorkload] === Batch {batch_num} ===")
            print(f"[FixedWorkload] Generating new workload for episodes {batch_num * self.batch_size}-{(batch_num + 1) * self.batch_size - 1}")
            
            # Generate new workload
            self._cached_workload = generator.generate(
                num_packets=num_packets,
                duration=duration
            )
            
            # Save batch workload if output_dir specified
            if self.output_dir:
                batch_path = Path(self.output_dir) / f"batch_{batch_num:04d}_workload.csv"
                self._cached_workload.to_csv(batch_path, index=False)
                print(f"[FixedWorkload] Saved to {batch_path}")
        else:
            ep_in_batch = episode_id % self.batch_size
            print(f"[FixedWorkload] Reusing batch {batch_num} workload (episode {ep_in_batch + 1}/{self.batch_size})")
        
        # Save to episode directory
        episode_workload_path = episode_dir / "workload.csv"
        self._cached_workload.to_csv(episode_workload_path, index=False)
        
        return episode_workload_path


# ============================================================================
# OPTION C: Minimal patch - just wrap the existing generate call
# ============================================================================

def make_fixed_workload_generator(original_generator, batch_size=10):
    """
    Wraps a workload generator to reuse workloads within batches.
    
    Usage:
        # In cloudsim_trainer.py, after creating workload_generator:
        from fixed_workload_patch import make_fixed_workload_generator
        workload_generator = make_fixed_workload_generator(workload_generator, batch_size=10)
    """
    
    class FixedWorkloadWrapper:
        def __init__(self, generator, batch_size):
            self.generator = generator
            self.batch_size = batch_size
            self._call_count = 0
            self._cached_workload = None
        
        def generate(self, num_packets, duration, **kwargs):
            if self._call_count % self.batch_size == 0:
                # Generate new workload at batch boundary
                print(f"[FixedWorkload] Generating new workload (batch {self._call_count // self.batch_size})")
                self._cached_workload = self.generator.generate(
                    num_packets=num_packets,
                    duration=duration,
                    **kwargs
                )
            else:
                print(f"[FixedWorkload] Reusing workload (call {self._call_count % self.batch_size + 1}/{self.batch_size})")
            
            self._call_count += 1
            return self._cached_workload.copy()
        
        # Forward other methods to original generator
        def __getattr__(self, name):
            return getattr(self.generator, name)
    
    return FixedWorkloadWrapper(original_generator, batch_size)


# ============================================================================
# Testing code
# ============================================================================

if __name__ == "__main__":
    # Test the FixedWorkloadManager
    print("Testing FixedWorkloadManager...")
    
    class MockGenerator:
        def __init__(self):
            self.call_count = 0
        
        def generate(self, num_packets, duration):
            self.call_count += 1
            import pandas as pd
            return pd.DataFrame({'packet': range(num_packets), 'gen_call': self.call_count})
    
    manager = FixedWorkloadManager(batch_size=3)
    generator = MockGenerator()
    
    for ep in range(10):
        print(f"\nEpisode {ep}:")
        workload = manager.get_workload(
            episode_id=ep,
            episode_dir="/tmp",
            generator=generator,
            num_packets=10,
            duration=1.0
        )
    
    print(f"\nGenerator was called {generator.call_count} times (expected: 4 for 10 episodes with batch_size=3)")