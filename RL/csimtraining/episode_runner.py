"""
episode_runner.py - Run CloudSim episodes as subprocesses.

This module handles:
1. Writing critical flows to file
2. Running CFRRLTrainingRunner as subprocess
3. Reading results (episode_summary.json, flow_summary.csv, link_stats.csv)
4. Error handling and timeouts
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


def compute_reward(
    episode_summary: Dict[str, Any],
    queuing_weight: float = 1.0,
    drop_penalty: float = 10.0,
    normalize_queuing: float = 1000.0,  # Convert ms to seconds
) -> float:
    """
    Compute reward from episode summary.
    
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


def parse_link_id(link_id: str) -> Tuple[str, str]:
    """Parse link ID like 'Switch: Denver->Switch: KansasCity' into (src, dst)."""
    parts = link_id.split('->')
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return link_id, link_id


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("EpisodeRunner test")
    print("==================")
    
    # Test reward computation
    test_summary = {
        'mean_queuing_ms': 14.06,
        'drop_rate': 0.0,
        'total_packets': 51
    }
    reward = compute_reward(test_summary)
    print(f"\nTest reward computation:")
    print(f"  Summary: {test_summary}")
    print(f"  Reward: {reward:.6f}")
    print(f"  Expected: {-14.06/1000 - 10*0:.6f}")
    
    # Test link ID parsing
    test_link = "Switch: Denver->Switch: KansasCity"
    src, dst = parse_link_id(test_link)
    print(f"\nLink parsing: '{test_link}'")
    print(f"  src: {src}")
    print(f"  dst: {dst}")