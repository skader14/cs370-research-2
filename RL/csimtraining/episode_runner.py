"""
episode_runner.py - Run CloudSim episodes as subprocesses.

This module handles:
1. Writing critical flows to file
2. Running CFRRLTrainingRunner as subprocess
3. Reading results (episode_summary.json, flow_summary.csv, link_stats.csv)
4. Computing dense rewards from per-packet latency data
5. Error handling and timeouts
"""

import subprocess
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import time
import os
import shutil


class EpisodeRunner:
    """
    Runs CloudSim training episodes as subprocesses.
    
    Usage:
        runner = EpisodeRunner(cloudsim_dir="path/to/cloudsimsdn")
        results = runner.run_episode(
            workload_file="episode_1_workload.csv",
            critical_flows=[84, 50, 109, 17, 23, 91, 42, 7],
            output_dir="outputs/episode_1/"
        )
    """
    
    def __init__(
        self,
        cloudsim_dir: str = ".",
        java_class: str = "org.cloudbus.cloudsim.sdn.example.CFRRLTrainingRunner",
        timeout: int = 120,  # seconds
        verbose: bool = True,
    ):
        """
        Initialize the episode runner.
        
        Args:
            cloudsim_dir: Path to CloudSimSDN project directory
            java_class: Full class name of training runner
            timeout: Maximum time to wait for episode (seconds)
            verbose: Print progress messages
        """
        self.cloudsim_dir = Path(cloudsim_dir)
        self.java_class = java_class
        self.timeout = timeout
        self.verbose = verbose
        
        # Verify directory exists
        if not self.cloudsim_dir.exists():
            raise ValueError(f"CloudSim directory not found: {cloudsim_dir}")
    
    def _log(self, msg: str) -> None:
        """Print message if verbose mode."""
        if self.verbose:
            print(f"[EpisodeRunner] {msg}")
    
    def write_critical_flows(self, flows: List[int], filepath: str) -> None:
        """Write critical flow IDs to file (one per line)."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            for flow_id in flows:
                f.write(f"{flow_id}\n")
    
    def run_episode(
        self,
        workload_file: str,
        critical_flows: List[int],
        output_dir: str,
        episode_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run a single training episode.
        
        Args:
            workload_file: Path to workload CSV (relative to cloudsim_dir)
            critical_flows: List of K flow IDs to optimize
            output_dir: Directory for output files (relative to cloudsim_dir)
            episode_id: Optional episode number for logging
        
        Returns:
            Dictionary containing:
                - success: bool
                - episode_summary: dict (from JSON)
                - flow_summary: DataFrame
                - link_stats: DataFrame
                - wall_time_ms: int
                - error: str (if failed)
        """
        start_time = time.time()
        episode_str = f"Episode {episode_id}" if episode_id is not None else "Episode"
        
        # Ensure output directory exists
        output_path = self.cloudsim_dir / output_dir
        output_path.mkdir(parents=True, exist_ok=True)
        
        # CloudSim creates a result directory based on workload path
        # e.g., workload at "test_setup/workload.csv" -> "result_test_setup/workload.csv"
        workload_path = Path(workload_file)
        if workload_path.parent.name:
            result_dir = self.cloudsim_dir / f"result_{workload_path.parent.name}"
            result_dir.mkdir(parents=True, exist_ok=True)
        
        # Write critical flows file
        critical_flows_file = output_path / "critical_flows.txt"
        self.write_critical_flows(critical_flows, str(critical_flows_file))
        
        self._log(f"{episode_str}: Starting (workload={workload_file}, k={len(critical_flows)})")
        
        # Build Maven command
        # On Windows, we need to build a single command string with proper quoting
        args_str = f"{workload_file} {critical_flows_file} {output_dir}"
        
        # Use command string instead of list for Windows compatibility
        cmd = f'mvn exec:java -q "-Dexec.mainClass={self.java_class}" "-Dexec.args={args_str}"'
        
        try:
            # Run CloudSim
            # shell=True needed on Windows to find mvn.cmd and handle quoting
            result = subprocess.run(
                cmd,
                cwd=str(self.cloudsim_dir),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                shell=True  # Required for Windows
            )
            
            wall_time_ms = int((time.time() - start_time) * 1000)
            
            # Check for errors
            if result.returncode != 0:
                self._log(f"{episode_str}: FAILED (return code {result.returncode})")
                return {
                    'success': False,
                    'error': f"Process returned {result.returncode}: {result.stderr}",
                    'wall_time_ms': wall_time_ms,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                }
            
            # Read results
            results = self._read_results(output_path)
            results['success'] = True
            results['wall_time_ms'] = wall_time_ms
            
            self._log(f"{episode_str}: Complete ({wall_time_ms}ms, "
                     f"mean_queue={results['episode_summary'].get('mean_queuing_ms', 0):.2f}ms)")
            
            return results
            
        except subprocess.TimeoutExpired:
            self._log(f"{episode_str}: TIMEOUT after {self.timeout}s")
            return {
                'success': False,
                'error': f"Timeout after {self.timeout} seconds",
                'wall_time_ms': self.timeout * 1000,
            }
        except Exception as e:
            self._log(f"{episode_str}: ERROR - {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'wall_time_ms': int((time.time() - start_time) * 1000),
            }
    
    def _read_results(self, output_dir: Path) -> Dict[str, Any]:
        """Read all result files from an episode."""
        results = {}
        
        # Episode summary (JSON)
        summary_file = output_dir / "episode_summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                results['episode_summary'] = json.load(f)
        else:
            results['episode_summary'] = {}
        
        # Flow summary (CSV)
        flow_file = output_dir / "flow_summary.csv"
        if flow_file.exists():
            results['flow_summary'] = pd.read_csv(flow_file)
        else:
            results['flow_summary'] = pd.DataFrame()
        
        # Link stats (CSV)
        link_file = output_dir / "link_stats.csv"
        if link_file.exists():
            results['link_stats'] = pd.read_csv(link_file)
        else:
            results['link_stats'] = pd.DataFrame()
        
        # Latency results (CSV) - for dense rewards
        latency_file = output_dir / "latency_results.csv"
        if latency_file.exists():
            results['latency_file'] = str(latency_file)
        else:
            results['latency_file'] = None
        
        return results
    
    def cleanup_episode(self, output_dir: str) -> None:
        """Remove episode output directory to save space."""
        output_path = self.cloudsim_dir / output_dir
        if output_path.exists():
            shutil.rmtree(output_path)
    
    def cleanup_cloudsim_results(self, pattern: str = "result_*") -> None:
        """
        Clean up CloudSim's auto-generated result directories.
        
        CloudSim creates 'result_<folder>' directories in the project root
        when processing workloads. These are typically empty or contain
        redundant data.
        """
        import glob
        for result_dir in glob.glob(str(self.cloudsim_dir / pattern)):
            result_path = Path(result_dir)
            if result_path.is_dir() and result_path.name.startswith("result_"):
                shutil.rmtree(result_path)


# =============================================================================
# REWARD COMPUTATION
# =============================================================================

def compute_reward(
    episode_summary: Dict[str, Any],
    queuing_weight: float = 1.0,
    drop_penalty: float = 10.0,
    normalize_queuing: float = 1000.0,  # Convert ms to seconds
) -> float:
    """
    Compute sparse reward from episode summary.
    
    Reward = -mean_queuing - drop_penalty * drop_rate
    
    Args:
        episode_summary: Dict from episode_summary.json
        queuing_weight: Weight on queuing delay
        drop_penalty: Penalty multiplier for drop rate
        normalize_queuing: Divide queuing by this (1000 to convert ms→s)
    
    Returns:
        Reward value (higher is better, typically negative)
    """
    mean_queuing_ms = episode_summary.get('mean_queuing_ms', 0)
    drop_rate = episode_summary.get('drop_rate', 0)
    
    # Normalize queuing to seconds for reasonable scale
    mean_queuing_s = mean_queuing_ms / normalize_queuing
    
    reward = -queuing_weight * mean_queuing_s - drop_penalty * drop_rate
    
    return reward


def compute_dense_rewards(
    latency_file: str,
    window_size: float = 1.0,
    queuing_weight: float = 1.0,
    drop_penalty: float = 10.0,
    gamma: float = 0.99,
    critical_only: bool = True,
    fallback_to_all: bool = True,
) -> Dict[str, Any]:
    """
    Compute dense rewards from per-packet latency data.
    
    Bins packets into time windows and computes per-window rewards.
    Returns both window-level rewards and the discounted total.
    
    Args:
        latency_file: Path to latency_results.csv from CloudSim
        window_size: Size of each time window in seconds (default: 1.0)
        queuing_weight: Weight on mean queuing delay
        drop_penalty: Penalty for drops (applied per window based on drop count)
        gamma: Discount factor for computing total reward (default: 0.99)
        critical_only: If True, compute rewards only on critical flow packets
        fallback_to_all: If True and no critical packets in window, use all packets
    
    Returns:
        Dictionary containing:
            'window_rewards': List[Tuple[float, float]] - (window_start_time, reward)
            'window_stats': List[Dict] - detailed stats per window
            'total_reward': float - discounted sum of window rewards
            'total_reward_undiscounted': float - simple sum of window rewards
            'num_windows': int
            'critical_packets': int
            'background_packets': int
            'total_packets': int
            'episode_duration': float
            'mean_window_reward': float
            'std_window_reward': float
    """
    # Read latency data
    try:
        df = pd.read_csv(latency_file)
    except Exception as e:
        return {
            'error': f"Failed to read latency file: {e}",
            'window_rewards': [],
            'total_reward': 0.0,
            'num_windows': 0,
        }
    
    if df.empty:
        return {
            'error': "Empty latency file",
            'window_rewards': [],
            'total_reward': 0.0,
            'num_windows': 0,
        }
    
    # Ensure required columns exist
    required_cols = ['timestamp', 'queuing_delay', 'is_critical']
    for col in required_cols:
        if col not in df.columns:
            return {
                'error': f"Missing required column: {col}",
                'window_rewards': [],
                'total_reward': 0.0,
                'num_windows': 0,
            }
    
    # Convert is_critical to boolean if needed
    if df['is_critical'].dtype == object:
        df['is_critical'] = df['is_critical'].str.lower() == 'true'
    
    # Get episode time range
    min_time = df['timestamp'].min()
    max_time = df['timestamp'].max()
    episode_duration = max_time - min_time
    
    # Calculate number of windows
    num_windows = max(1, int(np.ceil(episode_duration / window_size)))
    
    # Assign each packet to a window
    df['window'] = ((df['timestamp'] - min_time) / window_size).astype(int)
    df['window'] = df['window'].clip(upper=num_windows - 1)  # Handle edge case
    
    # Count packets
    critical_packets = df['is_critical'].sum()
    background_packets = len(df) - critical_packets
    
    # Compute per-window rewards
    window_rewards = []
    window_stats = []
    
    for w in range(num_windows):
        window_start = min_time + w * window_size
        window_data = df[df['window'] == w]
        
        # Filter to critical packets if requested
        if critical_only:
            critical_data = window_data[window_data['is_critical']]
            if len(critical_data) > 0:
                use_data = critical_data
            elif fallback_to_all and len(window_data) > 0:
                use_data = window_data  # Fallback to all packets
            else:
                use_data = pd.DataFrame()  # Empty
        else:
            use_data = window_data
        
        # Compute window statistics
        if len(use_data) > 0:
            mean_queuing = use_data['queuing_delay'].mean()
            max_queuing = use_data['queuing_delay'].max()
            packet_count = len(use_data)
            critical_count = use_data['is_critical'].sum() if 'is_critical' in use_data.columns else 0
            
            # Reward: negative queuing delay (already in seconds)
            # Note: queuing_delay in CSV is in seconds, not ms
            window_reward = -queuing_weight * mean_queuing
        else:
            # No packets in this window - neutral reward
            mean_queuing = 0.0
            max_queuing = 0.0
            packet_count = 0
            critical_count = 0
            window_reward = 0.0
        
        window_rewards.append((window_start, window_reward))
        window_stats.append({
            'window': w,
            'start_time': window_start,
            'end_time': window_start + window_size,
            'packet_count': packet_count,
            'critical_count': critical_count,
            'mean_queuing_s': mean_queuing,
            'max_queuing_s': max_queuing,
            'reward': window_reward,
        })
    
    # Compute total reward (discounted sum)
    rewards_only = [r for _, r in window_rewards]
    total_discounted = sum(
        (gamma ** t) * r for t, r in enumerate(rewards_only)
    )
    total_undiscounted = sum(rewards_only)
    
    # Statistics
    rewards_array = np.array(rewards_only)
    mean_reward = np.mean(rewards_array) if len(rewards_array) > 0 else 0.0
    std_reward = np.std(rewards_array) if len(rewards_array) > 0 else 0.0
    
    return {
        'window_rewards': window_rewards,
        'window_stats': window_stats,
        'total_reward': total_discounted,
        'total_reward_undiscounted': total_undiscounted,
        'num_windows': num_windows,
        'critical_packets': int(critical_packets),
        'background_packets': int(background_packets),
        'total_packets': len(df),
        'episode_duration': episode_duration,
        'mean_window_reward': mean_reward,
        'std_window_reward': std_reward,
        'window_size': window_size,
        'gamma': gamma,
    }


def compute_dense_reward_from_results(
    results: Dict[str, Any],
    window_size: float = 1.0,
    queuing_weight: float = 1.0,
    drop_penalty: float = 10.0,
    gamma: float = 0.99,
    critical_only: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function to compute dense rewards from episode results.
    
    Args:
        results: Dictionary returned by EpisodeRunner.run_episode()
        window_size: Size of time windows in seconds
        queuing_weight: Weight on queuing delay
        drop_penalty: Penalty for drops
        gamma: Discount factor
        critical_only: Use only critical flow packets
    
    Returns:
        Dense reward dictionary (same as compute_dense_rewards)
    """
    latency_file = results.get('latency_file')
    
    if latency_file is None:
        # Fall back to sparse reward
        sparse_reward = compute_reward(
            results.get('episode_summary', {}),
            queuing_weight=queuing_weight,
            drop_penalty=drop_penalty,
        )
        return {
            'error': "No latency file available, using sparse reward",
            'window_rewards': [(0.0, sparse_reward)],
            'total_reward': sparse_reward,
            'num_windows': 1,
            'fallback': True,
        }
    
    return compute_dense_rewards(
        latency_file=latency_file,
        window_size=window_size,
        queuing_weight=queuing_weight,
        drop_penalty=drop_penalty,
        gamma=gamma,
        critical_only=critical_only,
    )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def parse_link_id(link_id: str) -> Tuple[str, str]:
    """Parse link ID like 'Switch: Denver->Switch: KansasCity' into (src, dst)."""
    parts = link_id.split('->')
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return link_id, link_id


def compare_sparse_vs_dense(
    latency_file: str,
    episode_summary: Dict[str, Any],
    window_size: float = 1.0,
) -> Dict[str, Any]:
    """
    Compare sparse and dense reward computation for debugging/analysis.
    
    Args:
        latency_file: Path to latency_results.csv
        episode_summary: Dict from episode_summary.json
        window_size: Window size for dense rewards
    
    Returns:
        Dictionary with comparison statistics
    """
    sparse_reward = compute_reward(episode_summary)
    dense_result = compute_dense_rewards(latency_file, window_size=window_size)
    
    return {
        'sparse_reward': sparse_reward,
        'dense_reward_discounted': dense_result.get('total_reward', 0),
        'dense_reward_undiscounted': dense_result.get('total_reward_undiscounted', 0),
        'difference_discounted': dense_result.get('total_reward', 0) - sparse_reward,
        'num_windows': dense_result.get('num_windows', 0),
        'critical_packets': dense_result.get('critical_packets', 0),
        'total_packets': dense_result.get('total_packets', 0),
        'critical_ratio': (
            dense_result.get('critical_packets', 0) / dense_result.get('total_packets', 1)
            if dense_result.get('total_packets', 0) > 0 else 0
        ),
        'window_stats': dense_result.get('window_stats', []),
    }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("EpisodeRunner test")
    print("=" * 60)
    
    # Test sparse reward computation
    test_summary = {
        'mean_queuing_ms': 542.0,
        'drop_rate': 0.0,
        'total_packets': 1991
    }
    sparse_reward = compute_reward(test_summary)
    print(f"\nSparse reward computation:")
    print(f"  Summary: {test_summary}")
    print(f"  Reward: {sparse_reward:.6f}")
    print(f"  Expected: {-542.0/1000:.6f}")
    
    # Test dense reward computation (if latency file exists)
    test_latency_file = "latency_results.csv"
    if Path(test_latency_file).exists():
        print(f"\n" + "=" * 60)
        print("Dense reward computation test:")
        dense_result = compute_dense_rewards(test_latency_file, window_size=1.0)
        
        print(f"  Num windows: {dense_result.get('num_windows', 0)}")
        print(f"  Total packets: {dense_result.get('total_packets', 0)}")
        print(f"  Critical packets: {dense_result.get('critical_packets', 0)}")
        print(f"  Episode duration: {dense_result.get('episode_duration', 0):.2f}s")
        print(f"  Total reward (discounted): {dense_result.get('total_reward', 0):.6f}")
        print(f"  Total reward (undiscounted): {dense_result.get('total_reward_undiscounted', 0):.6f}")
        print(f"  Mean window reward: {dense_result.get('mean_window_reward', 0):.6f}")
        print(f"  Std window reward: {dense_result.get('std_window_reward', 0):.6f}")
        
        # Show per-window breakdown
        print(f"\n  Per-window breakdown:")
        for ws in dense_result.get('window_stats', [])[:5]:  # First 5 windows
            print(f"    Window {ws['window']}: t={ws['start_time']:.1f}-{ws['end_time']:.1f}s, "
                  f"packets={ws['packet_count']}, critical={ws['critical_count']}, "
                  f"mean_q={ws['mean_queuing_s']*1000:.1f}ms, r={ws['reward']:.4f}")
        
        if dense_result.get('num_windows', 0) > 5:
            print(f"    ... ({dense_result['num_windows'] - 5} more windows)")
    else:
        print(f"\n[Skipping dense reward test - {test_latency_file} not found]")
    
    # Test link ID parsing
    print(f"\n" + "=" * 60)
    test_link = "Switch: Denver->Switch: KansasCity"
    src, dst = parse_link_id(test_link)
    print(f"Link parsing: '{test_link}'")
    print(f"  src: {src}")
    print(f"  dst: {dst}")