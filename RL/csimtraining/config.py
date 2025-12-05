"""
config.py - Training configuration and hyperparameters.

This module centralizes all training configuration for easy modification.
"""

from dataclasses import dataclass, field
from typing import List
import json
from pathlib import Path


@dataclass
class TrainingConfig:
    """
    Training configuration for CloudSim-in-the-loop RL.
    
    All hyperparameters can be modified here or overridden via command line.
    """
    
    # ==========================================================================
    # Episode Settings
    # ==========================================================================
    
    num_episodes: int = 500           # Total training episodes
    packets_per_episode: int = 300    # Packets per workload
    episode_duration: float = 90.0    # Simulated seconds per episode
    
    # ==========================================================================
    # Network Topology
    # ==========================================================================
    
    num_nodes: int = 12               # Abilene nodes
    num_flows: int = 132              # 12 × 11 directed pairs
    k_critical: int = 8               # Flows to select for optimization
    
    # ==========================================================================
    # Feature Configuration
    # ==========================================================================
    
    num_features: int = 9             # Features per flow
    # Features: propagation_delay, path_length, bottleneck_capacity,
    #           demand, prev_mean_queuing, prev_max_queuing, prev_drop_rate,
    #           prev_path_utilization, prev_bottleneck_util
    
    # ==========================================================================
    # Policy Network Architecture
    # ==========================================================================
    
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    dropout: float = 0.1
    use_flow_wise: bool = False       # Use FlowWisePolicyNetwork instead
    
    # ==========================================================================
    # Training Hyperparameters
    # ==========================================================================
    
    learning_rate: float = 1e-4
    baseline_decay: float = 0.99      # EMA for REINFORCE baseline
    entropy_weight: float = 0.01      # Entropy bonus for exploration
    max_grad_norm: float = 1.0        # Gradient clipping
    
    # ==========================================================================
    # Temperature Schedule (for exploration)
    # ==========================================================================
    
    temperature_start: float = 1.0    # Initial temperature
    temperature_end: float = 0.1      # Final temperature
    temperature_decay: float = 0.995  # Per-episode decay
    
    # ==========================================================================
    # Reward Function
    # ==========================================================================
    
    queuing_weight: float = 1.0       # Weight on mean queuing delay
    drop_penalty: float = 10.0        # Penalty per drop rate unit
    # Reward = -queuing_weight * (mean_queuing_ms / 1000) - drop_penalty * drop_rate
    
    # ==========================================================================
    # Workload Generation
    # ==========================================================================
    
    min_packet_size: int = 1_000_000        # 1 MB
    max_packet_size: int = 500_000_000      # 500 MB
    flow_activity_prob: float = 0.4         # Probability each flow is active
    burst_prob: float = 0.1                 # Probability of burst traffic
    burst_multiplier: float = 3.0           # Burst size multiplier
    
    # ==========================================================================
    # Checkpointing and Logging
    # ==========================================================================
    
    checkpoint_freq: int = 50         # Save checkpoint every N episodes
    eval_freq: int = 10               # Log evaluation metrics every N episodes
    keep_episode_data: int = 100      # Keep episode data every N episodes
    
    # ==========================================================================
    # Paths
    # ==========================================================================
    
    cloudsim_dir: str = "."
    output_dir: str = "training_outputs"
    
    # ==========================================================================
    # Methods
    # ==========================================================================
    
    def save(self, filepath: str) -> None:
        """Save config to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'TrainingConfig':
        """Load config from JSON file."""
        with open(filepath) as f:
            data = json.load(f)
        return cls(**data)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'num_episodes': self.num_episodes,
            'packets_per_episode': self.packets_per_episode,
            'episode_duration': self.episode_duration,
            'num_nodes': self.num_nodes,
            'num_flows': self.num_flows,
            'k_critical': self.k_critical,
            'num_features': self.num_features,
            'hidden_dims': self.hidden_dims,
            'dropout': self.dropout,
            'use_flow_wise': self.use_flow_wise,
            'learning_rate': self.learning_rate,
            'baseline_decay': self.baseline_decay,
            'entropy_weight': self.entropy_weight,
            'max_grad_norm': self.max_grad_norm,
            'temperature_start': self.temperature_start,
            'temperature_end': self.temperature_end,
            'temperature_decay': self.temperature_decay,
            'queuing_weight': self.queuing_weight,
            'drop_penalty': self.drop_penalty,
            'min_packet_size': self.min_packet_size,
            'max_packet_size': self.max_packet_size,
            'flow_activity_prob': self.flow_activity_prob,
            'burst_prob': self.burst_prob,
            'burst_multiplier': self.burst_multiplier,
            'checkpoint_freq': self.checkpoint_freq,
            'eval_freq': self.eval_freq,
            'keep_episode_data': self.keep_episode_data,
            'cloudsim_dir': self.cloudsim_dir,
            'output_dir': self.output_dir,
        }
    
    def __str__(self) -> str:
        """Pretty print configuration."""
        lines = ["TrainingConfig:"]
        for key, value in self.to_dict().items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)


# =============================================================================
# Preset Configurations
# =============================================================================

def get_fast_debug_config() -> TrainingConfig:
    """Fast configuration for debugging (10 episodes, small workloads)."""
    return TrainingConfig(
        num_episodes=10,
        packets_per_episode=50,
        episode_duration=30.0,
        checkpoint_freq=5,
        output_dir="debug_outputs",
    )


def get_standard_config() -> TrainingConfig:
    """Standard training configuration (500 episodes)."""
    return TrainingConfig()


def get_long_training_config() -> TrainingConfig:
    """Extended training configuration (1000 episodes, larger workloads)."""
    return TrainingConfig(
        num_episodes=1000,
        packets_per_episode=400,
        episode_duration=120.0,
        temperature_decay=0.997,  # Slower decay
        output_dir="long_training_outputs",
    )


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("Configuration Presets")
    print("=" * 50)
    
    print("\n1. Debug Config:")
    debug_config = get_fast_debug_config()
    print(debug_config)
    
    print("\n2. Standard Config:")
    standard_config = get_standard_config()
    print(standard_config)
    
    # Save and load test
    standard_config.save("test_config.json")
    loaded_config = TrainingConfig.load("test_config.json")
    print(f"\nConfig save/load test: {'PASS' if loaded_config.num_episodes == 500 else 'FAIL'}")
    
    Path("test_config.json").unlink()  # Cleanup