#!/usr/bin/env python3
"""
CFR-RL Agent for CloudSimSDN Integration

With comprehensive logging for debugging.
All logs go to stderr (stdout is reserved for JSON protocol).
Also writes to cfrrl_agent.log file.
"""

import sys
import json
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# ==============================================================================
# MODEL CONFIGURATION - EDIT THIS SECTION TO SWITCH MODELS
# ==============================================================================

# Uncomment ONE of these lines to select which model to use:

# MODEL_PATH = "best_abilene_lw0.0.pt"   # MLU-only (original CFR-RL)
MODEL_PATH = "best_abilene_lw0.3.pt"   # Latency-aware (latency_weight=0.3)
# MODEL_PATH = "best_abilene_v2.pt"      # Legacy model (if you have one)

# Log file name (change this when switching models for clearer logs)
# LOG_FILE = "cfrrl_agent_lw0.0.log"      # Match with model above
LOG_FILE = "cfrrl_agent_lw0.3.log"
# LOG_FILE = "cfrrl_agent.log"

# ==============================================================================
# END CONFIGURATION
# ==============================================================================


log_file_handle = None

def init_logging():
    """Initialize logging to file."""
    global log_file_handle
    try:
        log_file_handle = open(LOG_FILE, 'w')
        log("INIT", f"Agent log started at {datetime.now()}")
        log("INIT", f"Log file: {os.path.abspath(LOG_FILE)}")
        log("INIT", f"Model: {MODEL_PATH}")
        log("INIT", f"Python version: {sys.version}")
        log("INIT", f"Working directory: {os.getcwd()}")
    except Exception as e:
        print(f"[Agent] WARNING: Could not create log file: {e}", file=sys.stderr)

def log(component: str, message: str, level: str = "INFO"):
    """Log a message to stderr and file."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    log_line = f"[{timestamp}] [{level:5s}] [{component:12s}] {message}"
    
    # Always write to stderr (visible to Java)
    print(log_line, file=sys.stderr)
    
    # Also write to file
    if log_file_handle:
        log_file_handle.write(log_line + "\n")
        log_file_handle.flush()

def log_json(component: str, label: str, data: Any):
    """Log JSON data (truncated for readability)."""
    json_str = json.dumps(data) if not isinstance(data, str) else data
    if len(json_str) > 300:
        display = json_str[:300] + f"... [{len(json_str)} chars]"
    else:
        display = json_str
    log(component, f"{label}: {display}", "DEBUG")
    
    # Write full JSON to file
    if log_file_handle and len(json_str) > 300:
        log_file_handle.write(f"    [FULL JSON]: {json_str}\n")
        log_file_handle.flush()

def close_logging():
    """Close log file."""
    global log_file_handle
    if log_file_handle:
        log("INIT", f"Agent log ended at {datetime.now()}")
        log_file_handle.close()
        log_file_handle = None


# ==================== TRAINING CONFIG (for model loading) ====================

@dataclass
class TrainingConfig:
    """Training configuration - must match the one used during training."""
    k_critical: int = 8
    hidden_dim: int = 64
    
    total_iterations: int = 5000
    lr: float = 1e-3
    entropy_coef: float = 0.02
    grad_clip: float = 1.0
    
    temperature: float = 1.0
    temperature_decay: float = 0.999
    min_temperature: float = 0.3
    
    eval_interval: int = 500
    eval_episodes: int = 50
    log_interval: int = 100
    
    early_stop_patience: int = 3
    
    # New field for latency-aware models
    latency_weight: float = 0.0


# ==================== PYTORCH SETUP ====================

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    log("INIT", "PyTorch available: YES")
except ImportError:
    TORCH_AVAILABLE = False
    log("INIT", "PyTorch available: NO - will use Top-K fallback", "WARN")


# ==================== MODEL DEFINITION ====================

if TORCH_AVAILABLE:
    class StablePointwisePolicy(nn.Module):
        """Pointwise scoring network."""
        
        def __init__(self, num_flows: int, hidden_dim: int = 64):
            super().__init__()
            self.num_flows = num_flows
            
            self.flow_encoder = nn.Sequential(
                nn.Linear(4, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )
            
            self.global_encoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            
            self.score_head = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            local_features = self.flow_encoder(x)
            global_context = local_features.mean(dim=1, keepdim=True)
            global_context = self.global_encoder(global_context)
            global_context = global_context.expand(-1, x.shape[1], -1)
            combined = torch.cat([local_features, global_context], dim=-1)
            scores = self.score_head(combined).squeeze(-1)
            return scores


# ==================== AGENT ====================

class CFRRLAgent:
    """Agent that uses trained CFR-RL model to select critical flows."""
    
    def __init__(self, model_path: str = MODEL_PATH):
        self.model = None
        self.model_num_flows = 132
        self.update_count = 0
        self.model_latency_weight = None  # Will be loaded from checkpoint if available
        
        log("Agent", "Initializing CFR-RL Agent")
        log("Agent", f"Model path: {model_path}")
        
        if TORCH_AVAILABLE:
            self._load_model(model_path)
        else:
            log("Agent", "No PyTorch - will use Top-K fallback", "WARN")
    
    def _load_model(self, model_path: str):
        """Load trained model."""
        paths_to_try = [
            model_path,
            os.path.join(os.path.dirname(__file__), model_path),
            os.path.join("RL", model_path),
            os.path.join("..", model_path),
        ]
        
        log("Agent", f"Searching for model in {len(paths_to_try)} locations...")
        
        for path in paths_to_try:
            abs_path = os.path.abspath(path)
            exists = os.path.exists(path)
            log("Agent", f"  Trying: {abs_path} - exists: {exists}", "DEBUG")
            
            if exists:
                try:
                    log("Agent", f"Loading model from: {abs_path}")
                    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
                    
                    # Load model architecture
                    self.model = StablePointwisePolicy(num_flows=132, hidden_dim=64)
                    self.model.load_state_dict(checkpoint['policy_state_dict'])
                    self.model.eval()
                    
                    # Check for latency_weight in checkpoint
                    if 'latency_weight' in checkpoint:
                        self.model_latency_weight = checkpoint['latency_weight']
                        log("Agent", f"Model latency_weight: {self.model_latency_weight}")
                    else:
                        log("Agent", "Model latency_weight: not saved (legacy model)")
                    
                    log("Agent", "Model loaded successfully!")
                    log("Agent", f"Model num_flows: {self.model.num_flows}")
                    return
                except Exception as e:
                    log("Agent", f"Failed to load {path}: {e}", "ERROR")
        
        log("Agent", "No model found - will use Top-K fallback", "WARN")
    
    def select_critical_flows(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Select K critical flows from the current state."""
        self.update_count += 1
        
        log("Agent", f"========== UPDATE #{self.update_count} ==========")
        
        flows = state.get('flows', [])
        k = state.get('k', 8)
        sim_time = state.get('time', 'N/A')
        
        log("Agent", f"Simulation time: {sim_time}")
        log("Agent", f"Received {len(flows)} flows, K={k}")
        
        if not flows:
            log("Agent", "No flows - returning empty selection", "WARN")
            return {
                'critical_flows': [],
                'k': k,
                'method': 'empty',
                'update_num': self.update_count
            }
        
        # Log flow details
        log("Agent", "Flow details:", "DEBUG")
        for i, flow in enumerate(flows[:10]):  # Limit to first 10
            flow_id = flow.get('id', 'N/A')
            bw = flow.get('bw', 0)
            features = flow.get('features', [])
            log("Agent", f"  Flow {flow_id}: bw={bw}, features={features}", "DEBUG")
        if len(flows) > 10:
            log("Agent", f"  ... and {len(flows) - 10} more flows", "DEBUG")
        
        # Try model selection
        if self.model is not None:
            try:
                log("Agent", "Using CFR-RL model for selection")
                start_time = time.time()
                critical_ids = self._select_with_model(flows, k)
                elapsed = (time.time() - start_time) * 1000
                log("Agent", f"Model inference took {elapsed:.1f}ms")
                log("Agent", f"Selected critical flows: {critical_ids}")
                
                return {
                    'critical_flows': critical_ids,
                    'k': k,
                    'method': 'cfr_rl',
                    'model': MODEL_PATH,
                    'latency_weight': self.model_latency_weight,
                    'update_num': self.update_count
                }
            except Exception as e:
                log("Agent", f"Model selection failed: {e}", "ERROR")
                log("Agent", "Falling back to Top-K", "WARN")
        
        # Fallback to Top-K
        log("Agent", "Using Top-K fallback")
        critical_ids = self._select_top_k(flows, k)
        log("Agent", f"Top-K selected: {critical_ids}")
        
        return {
            'critical_flows': critical_ids,
            'k': k,
            'method': 'top_k',
            'update_num': self.update_count
        }
    
    def _select_with_model(self, flows: List[Dict], k: int) -> List[int]:
        """Use trained model to score and select flows."""
        features_list = []
        flow_ids = []
        
        for flow in flows:
            flow_id = flow.get('id', -1)
            if flow_id == -1:
                continue
            flow_ids.append(flow_id)
            
            if 'features' in flow and len(flow['features']) == 4:
                features_list.append(flow['features'])
            else:
                bw = flow.get('bw', 1e8)
                path_len = flow.get('path_len', 3)
                bottleneck = flow.get('bottleneck', 10e9)
                num_paths = flow.get('num_paths', 2)
                
                max_demand = 1e9
                max_cap = 10e9
                
                features_list.append([
                    min(bw / max_demand, 1.0),
                    min(num_paths / 4.0, 1.0),
                    min(path_len / 5.0, 1.0),
                    min(bottleneck / max_cap, 1.0)
                ])
        
        if not features_list:
            return []
        
        num_actual_flows = len(features_list)
        log("Agent", f"Building feature matrix: {num_actual_flows} flows x 4 features", "DEBUG")
        
        # Pad or truncate
        if num_actual_flows < self.model_num_flows:
            padding = [[0, 0, 0, 0]] * (self.model_num_flows - num_actual_flows)
            features_list = features_list + padding
        elif num_actual_flows > self.model_num_flows:
            sorted_indices = sorted(range(len(flow_ids)), 
                                   key=lambda i: features_list[i][0], reverse=True)
            sorted_indices = sorted_indices[:self.model_num_flows]
            features_list = [features_list[i] for i in sorted_indices]
            flow_ids = [flow_ids[i] for i in sorted_indices]
            num_actual_flows = self.model_num_flows
        
        # Model inference
        with torch.no_grad():
            x = torch.FloatTensor(features_list).unsqueeze(0)
            scores = self.model(x).squeeze(0).numpy()
        
        scores = scores[:num_actual_flows]
        
        # Log scores
        log("Agent", "Flow scores:", "DEBUG")
        for i, (fid, score) in enumerate(zip(flow_ids, scores)):
            if i < 10:  # Limit logging
                log("Agent", f"  Flow {fid}: score={score:.4f}", "DEBUG")
        
        # Select top K
        k = min(k, num_actual_flows)
        top_indices = scores.argsort()[-k:][::-1]
        critical_ids = [flow_ids[i] for i in top_indices]
        
        return critical_ids
    
    def _select_top_k(self, flows: List[Dict], k: int) -> List[int]:
        """Fallback: select top K flows by bandwidth."""
        def get_demand(flow):
            if 'features' in flow and len(flow['features']) >= 1:
                return flow['features'][0]
            return flow.get('bw', 0)
        
        sorted_flows = sorted(flows, key=get_demand, reverse=True)
        k = min(k, len(sorted_flows))
        critical_ids = [f.get('id', -1) for f in sorted_flows[:k] if f.get('id', -1) != -1]
        
        return critical_ids


# ==================== MAIN LOOP ====================

def main():
    """Main loop: read JSON from stdin, process, write JSON to stdout."""
    
    init_logging()
    
    log("MAIN", "=" * 60)
    log("MAIN", "CFR-RL AGENT STARTED")
    log("MAIN", "=" * 60)
    log("MAIN", f"Using model: {MODEL_PATH}")
    log("MAIN", "Protocol: Read JSON from stdin, write JSON to stdout")
    log("MAIN", "Shutdown: Send {\"shutdown\": true}")
    
    agent = CFRRLAgent()
    
    log("MAIN", "Ready, waiting for input...")
    log("MAIN", "-" * 60)
    
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        
        log("MAIN", f"Received input: {len(line)} chars")
        log_json("MAIN", "Input", line)
        
        try:
            state = json.loads(line)
            
            if state.get('shutdown', False):
                log("MAIN", "Received shutdown signal")
                break
            
            # Process and respond
            response = agent.select_critical_flows(state)
            response_json = json.dumps(response)
            
            log_json("MAIN", "Output", response_json)
            
            # Write to stdout (this is what Java reads!)
            print(response_json, flush=True)
            
            log("MAIN", f"Response sent: {len(response_json)} chars")
            log("MAIN", "-" * 60)
            
        except json.JSONDecodeError as e:
            log("MAIN", f"JSON parse error: {e}", "ERROR")
            error_response = json.dumps({'error': str(e), 'critical_flows': []})
            print(error_response, flush=True)
            
        except Exception as e:
            log("MAIN", f"Error: {e}", "ERROR")
            import traceback
            traceback.print_exc(file=sys.stderr)
            error_response = json.dumps({'error': str(e), 'critical_flows': []})
            print(error_response, flush=True)
    
    log("MAIN", "=" * 60)
    log("MAIN", f"CFR-RL AGENT EXITING (processed {agent.update_count} updates)")
    log("MAIN", "=" * 60)
    
    close_logging()


if __name__ == "__main__":
    main()