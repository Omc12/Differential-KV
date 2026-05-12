import torch
import numpy as np
from typing import Dict, List
from analysis.cross_layer_resonance import CrossLayerResonanceAnalyzer, ResonanceMetrics
from analysis.drift_tensor_analysis import DriftTensorAnalyzer
from anchor_logic.layer_coupling_graph import LayerCouplingGraph
from .resonance_controller import ResonanceController

class GlobalSyncManager:
    """
    Orchestrates resonance across all layers.
    Monitors global coherence and regulates layer coupling.
    """
    def __init__(self, num_layers: int, hidden_dim: int):
        self.num_layers = num_layers
        self.analyzer = CrossLayerResonanceAnalyzer(num_layers)
        self.drift_analyzer = DriftTensorAnalyzer(num_layers)
        self.coupling_graph = LayerCouplingGraph(num_layers)
        self.controllers = [ResonanceController(i, hidden_dim) for i in range(num_layers)]
        
        self.sync_history = []
        self.collapse_threshold = 0.4
        
    def step(self, layer_hidden_states: List[torch.Tensor]):
        """
        Main synchronization step. 
        1. Record states
        2. Analyze resonance
        3. Update coupling
        4. Apply stabilization
        """
        # 1. Update analysis
        for i, h in enumerate(layer_hidden_states):
            self.analyzer.add_state(i, h)
            
        metrics = self.analyzer.compute_resonance()
        self.sync_history.append(metrics.coherence_score)
        
        # 2. Update coupling graph
        self.coupling_graph.update_coupling(metrics.alignment_matrix)
        
        # 3. Detect and handle instability
        if metrics.coherence_score < self.collapse_threshold:
            self._apply_global_restoration(metrics)
            
        return metrics

    def _apply_global_restoration(self, metrics: ResonanceMetrics):
        """Increases restoration strength across all controllers when sync is low."""
        boost = (self.collapse_threshold - metrics.coherence_score) * 2.0
        for controller in self.controllers:
            # Dynamically increase restoration strength
            new_strength = min(0.3, 0.1 + boost)
            controller.set_strength(new_strength)

    def get_sync_telemetry(self) -> Dict:
        return {
            "coherence": self.sync_history[-1] if self.sync_history else 0.0,
            "entropy": self.analyzer.compute_resonance().synchronization_entropy,
            "is_stable": self.sync_history[-1] > self.collapse_threshold if self.sync_history else True,
            "sync_graph_connectivity": self.coupling_graph.get_global_sync_state()
        }
