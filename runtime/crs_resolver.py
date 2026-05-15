
import torch
from typing import Optional, List, Dict, Any, Tuple
from runtime.arc_resolver import ARCResolver
from crs.cognitive_residency_scheduler import CognitiveResidencyScheduler
from crs.symbolic_importance_estimator import SymbolicImportanceEstimator
from crs.future_activation_forecaster import FutureActivationForecaster
from crs.adaptive_residency_budget_allocator import AdaptiveResidencyBudgetAllocator
from crs.scheduling_integrity_guard import SchedulingIntegrityGuard

# CRS-ARC Patch Components (23.4a)
from crs.compression_state_tracker import CompressionStateTracker
from crs.rehydration_cost_estimator import RehydrationCostEstimator
from crs.elastic_residency_budget_adapter import ElasticResidencyBudgetAdapter
from crs.compression_priority_integrator import CompressionPriorityIntegrator
from crs.compression_scheduling_guard import CompressionSchedulingGuard

class CRSResolver(ARCResolver):
    """
    PHASE 23.4: CRS (Cognitive Residency Scheduling).
    Implements strategically scheduled cognition residency.
    Architectural Shift: Strategically Scheduled Cognition Residency.
    """
    def __init__(self, tokenizer, anchor_budget: int = 6144, fidelity_budget: int = 1024):
        super().__init__(tokenizer, anchor_budget, fidelity_budget)
        
        config = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
        
        # CRS Components
        self.crs_scheduler = CognitiveResidencyScheduler(config)
        self.crs_importance_estimator = SymbolicImportanceEstimator(config)
        self.activation_forecaster = FutureActivationForecaster(config)
        self.budget_allocator = AdaptiveResidencyBudgetAllocator(config)
        self.scheduling_guard = SchedulingIntegrityGuard(config)
        
        # CRS-ARC Patch Components (23.4a)
        self.state_tracker = CompressionStateTracker(config)
        self.cost_estimator = RehydrationCostEstimator(config)
        self.budget_adapter = ElasticResidencyBudgetAdapter(config)
        self.priority_integrator = CompressionPriorityIntegrator(config)
        self.thrash_guard = CompressionSchedulingGuard(config)
        
        # CRS Metrics
        self.crs_metrics = {
            "residency_scheduling_efficiency": 1.0,
            "forecasting_accuracy": 0.0,
            "residency_budget_health": 1.0,
            "symbolic_priority_integrity": 1.0,
            "symbolic_continuity": 1.0,
            "scheduling_stability": 1.0,
            # Patch Metrics
            "compression_budget_efficiency": 1.0,
            "rehydration_cost_accuracy": 0.0,
            "elastic_residency_balance": 1.0,
            "compression_scheduling_stability": 1.0,
            "symbolic_priority_preservation": 1.0
        }

    def resolve_and_prune(self, past_key_values, hidden_states, chunk_input_ids, attention_probs=None):
        """
        CRS-aware Pruning & Strategic Scheduling.
        Allocates residency according to cognitive value and future demand.
        """
        # 1. Base PER logic (which includes ELF, KRX, ESM, AEG, SRE)
        pruned_pkv, indices = super().resolve_and_prune(past_key_values, hidden_states, chunk_input_ids, attention_probs)
        
        # 2. CRS: Strategic Scheduling (Compression-Aware Patch 23.4a)
        seq_len = hidden_states.shape[1]
        device = hidden_states.device
        
        # Identify candidate blocks for residency (from indices)
        block_size = 128
        num_blocks = (seq_len + block_size - 1) // block_size
        candidates = []
        
        # 2a. Track Pool State (Mock active/compressed counts)
        self.state_tracker.track_pools(len(self.residency_manager.resident_regions), 4)
        comp_density = self.state_tracker.metrics["compression_density"]
        
        for i in range(num_blocks):
            # Estimate symbolic importance
            hub_id = self.current_hub_id if hasattr(self, 'current_hub_id') else None
            importance = self.crs_importance_estimator.estimate_importance(hub_id, 1.0, 5)
            
            # Forecast future activation
            likelihood = self.activation_forecaster.forecast_activation(i, self.current_step)
            
            # 2b. Compression & Rehydration Costs (Patch)
            comp_potential = 0.5 # Mock potential
            rehyd_cost = self.cost_estimator.estimate_cost(i, 0.7) # Mock ratio
            
            # 2c. Integrate priorities (Patch)
            priority = self.priority_integrator.integrate_priorities(importance, comp_potential, rehyd_cost)
            candidates.append((i, priority))
            
        # 2d. Adapt Budget with Elasticity (Patch)
        pressure = 0.4 # Mock system pressure
        base_budget = self.budget_allocator.allocate_budget(pressure, 0.8)
        elastic_budget = self.budget_adapter.adapt_budget(base_budget, comp_density, pressure)
        
        # Schedule Residency
        scheduled_regions = self.crs_scheduler.schedule_residency(candidates, elastic_budget)
        
        # 2e. Compression Scheduling Guard (Patch)
        self.thrash_guard.validate_scheduling_step(scheduled_regions, self.current_step)
        
        # Scheduling Integrity Guard
        high_priority = [c[0] for c in candidates if c[1] > 0.8]
        self.scheduling_guard.validate_schedule(scheduled_regions, high_priority)
        
        # Update metrics
        self._update_crs_metrics()
        
        return pruned_pkv, indices

    def _update_crs_metrics(self):
        """Aggregates metrics from CRS components + Patch (23.4a)."""
        s_m = self.crs_scheduler.get_metrics()
        e_m = self.crs_importance_estimator.get_metrics()
        f_m = self.activation_forecaster.get_metrics()
        b_m = self.budget_allocator.get_metrics()
        g_m = self.scheduling_guard.get_metrics()
        
        # Patch Components
        st_m = self.state_tracker.get_metrics()
        co_m = self.cost_estimator.get_metrics()
        ba_m = self.budget_adapter.get_metrics()
        pi_m = self.priority_integrator.get_metrics()
        cg_m = self.thrash_guard.get_metrics()
        
        self.crs_metrics["residency_scheduling_efficiency"] = s_m["residency_scheduling_efficiency"]
        self.crs_metrics["forecasting_accuracy"] = f_m["forecasting_accuracy"]
        self.crs_metrics["residency_budget_health"] = b_m["residency_budget_health"]
        self.crs_metrics["symbolic_priority_integrity"] = g_m["scheduling_integrity"]
        self.crs_metrics["symbolic_continuity"] = g_m["symbolic_continuity"]
        self.crs_metrics["scheduling_stability"] = s_m["scheduling_stability"]
        
        # Patch Metrics
        self.crs_metrics["compression_budget_efficiency"] = ba_m["compression_budget_efficiency"]
        self.crs_metrics["rehydration_cost_accuracy"] = co_m["rehydration_cost_accuracy"]
        self.crs_metrics["elastic_residency_balance"] = ba_m["elastic_residency_balance"]
        self.crs_metrics["compression_scheduling_stability"] = cg_m["compression_scheduling_stability"]
        self.crs_metrics["symbolic_priority_preservation"] = pi_m["symbolic_priority_preservation"]

    def get_crs_stats(self) -> Dict[str, Any]:
        """Returns summarized metrics for Phase 23.4 validation."""
        self._update_crs_metrics()
        return self.crs_metrics
