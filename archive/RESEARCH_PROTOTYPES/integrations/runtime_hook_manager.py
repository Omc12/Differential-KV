"""
integrations/runtime_hook_manager.py

Central manager for intercepting KV cache and applying DKV stabilization
during real-world inference across different runtimes.
"""

import torch
import time
from typing import Dict, Any, Optional, List
from runtime.unified_cognitive_runtime import UnifiedCognitiveRuntime
from runtime.adaptive_resonance_scheduler import AdaptiveResonanceScheduler

class RuntimeHookManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.runtime = UnifiedCognitiveRuntime(config)
        self.resonance_scheduler = AdaptiveResonanceScheduler(config)
        
        self.telemetry = {
            "token_count": 0,
            "intervention_count": 0,
            "total_latency_ms": 0.0,
            "manifold_drift": [],
            "regime_transitions": []
        }
        
    def on_token_start(self, token_id: int):
        """Called before processing a new token."""
        self.telemetry["token_count"] += 1
        
    def intercept_kv(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Intercepts KV tensors from the runtime.
        Returns the stabilized/compressed KV if applicable.
        """
        start_time = time.time()
        
        # 1. Update cognitive state
        state = self.runtime.state_engine.update(layer_idx, k, v)
        
        # 2. Check for resonance pulse
        should_pulse = self.resonance_scheduler.should_pulse(state)
        
        if should_pulse:
            self.telemetry["intervention_count"] += 1
            # Apply stabilization (logic from ACTR/GRP/SAM)
            # For now, we simulate the hook effect on the real tensor
            # In real integration, this might modify the tensor in-place or return a new one
            k, v = self.runtime.apply_stabilization(layer_idx, k, v, state)
            
        latency = (time.time() - start_time) * 1000
        self.telemetry["total_latency_ms"] += latency
        
        return k, v

    def on_generation_step(self, hidden_states: torch.Tensor):
        """
        Monitors hidden states for manifold tracking and regime classification.
        """
        # Feature extraction for regime classification
        features = self._extract_trajectory_features(hidden_states)
        regime = self.runtime.classify_regime(features)
        
        if regime != self.runtime.current_regime:
            self.telemetry["regime_transitions"].append({
                "token": self.telemetry["token_count"],
                "from": self.runtime.current_regime,
                "to": regime
            })
            self.runtime.update_regime(regime)

    def _extract_trajectory_features(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Simplified feature extraction (to be improved in Task 5)
        # hidden_states: [layers, batch, seq, dim]
        mean = hidden_states.mean(dim=-1)
        std = hidden_states.std(dim=-1)
        return torch.cat([mean.flatten(), std.flatten()])

    def get_telemetry_report(self) -> Dict[str, Any]:
        return {
            "avg_latency_per_token": self.telemetry["total_latency_ms"] / max(1, self.telemetry["token_count"]),
            "intervention_density": self.telemetry["intervention_count"] / max(1, self.telemetry["token_count"]),
            "regime_switches": len(self.telemetry["regime_transitions"]),
            "total_tokens": self.telemetry["token_count"]
        }
