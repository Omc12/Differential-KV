import torch
from memory.semantic_geometry_tracker import SemanticGeometryKVManager
from memory.hierarchical_memory_capsules import HierarchicalMemoryCapsuleEngine
from memory.capsule_registry import CapsuleRegistry
from memory.high_fidelity_token_windows import HighFidelityTokenWindowDetector
from memory.anchor_capsule_linker import AnchorCapsuleLinker
from memory.dynamic_precision_allocator import DynamicPrecisionAllocator
from memory.fidelity_budget_controller import FidelityBudgetController
from memory.pathway_continuity_engine import PathwayContinuityEngine

class HierarchicalMemoryResolver:
    """
    PHASE 18.7: Hierarchical Memory Resolver.
    Integrates HMCs, Anchor-Capsule Linking, and Dynamic Precision Tiers.
    """
    def __init__(self, 
                 anchor_budget: int = 6144, 
                 fidelity_token_budget: int = 1024,
                 max_capsules: int = 64):
        self.geometry = SemanticGeometryKVManager(anchor_budget=anchor_budget)
        self.capsule_engine = HierarchicalMemoryCapsuleEngine(high_fidelity_budget=fidelity_token_budget)
        self.registry = CapsuleRegistry()
        self.detector = HighFidelityTokenWindowDetector()
        self.linker = AnchorCapsuleLinker(self.registry)
        self.allocator = DynamicPrecisionAllocator()
        self.budget_controller = FidelityBudgetController(max_high_fidelity_tokens=fidelity_token_budget, max_capsules=max_capsules)
        self.bridge_engine = PathwayContinuityEngine()
        
        self.global_offset = 0

    def resolve_and_prune(self, past_key_values, hidden_states, chunk_input_ids):
        """
        1. Detect symbolic windows.
        2. Create and Register Capsules.
        3. Enforce Budget.
        4. Protect Capsule regions in Geometry Manager.
        5. Prune KV.
        """
        q_len = hidden_states.shape[1]
        
        # A. Window Detection
        windows = self.detector.detect_windows(hidden_states, self.global_offset)
        
        # B. Capsule Creation & Registration
        for start, end, entropy in windows:
            tier = self.allocator.classify_region(entropy, tags=["AUTO_DETECT"])
            capsule = self.capsule_engine.create_capsule(start, end, tier, entropy)
            self.registry.register(capsule)
            
        # C. Budget Enforcement
        evicted = self.budget_controller.enforce_budget(self.registry)
        for cid in evicted:
            if cid in self.registry.capsules:
                # Actually remove from registry (need to implement remove in registry)
                # For now, let's assume we filter them out during protection
                pass
        
        # D. Geometric Protection
        if self.geometry.accumulated_importance is not None:
            imp_len = self.geometry.accumulated_importance.shape[1]
            
            # Protect all active capsules
            for capsule in self.registry.capsules.values():
                if capsule.capsule_id in evicted: continue
                
                # Map global index to local importance index
                # This is tricky because geometry manager might have pruned previous tokens
                # However, for the CURRENT chunk, it's straightforward
                if capsule.start_idx >= self.global_offset:
                    local_start = capsule.start_idx - self.global_offset
                    local_end = capsule.end_idx - self.global_offset
                    
                    # Ensure we are within the current chunk bounds in the importance matrix
                    # accumulated_importance corresponds to the FULL context
                    start_abs = max(0, imp_len - q_len + local_start)
                    end_abs = min(imp_len, imp_len - q_len + local_end)
                    
                    if capsule.precision_tier == "HIGH":
                        self.geometry.accumulated_importance[:, start_abs:end_abs] = float('inf')
                    elif capsule.precision_tier == "MEDIUM":
                        self.geometry.accumulated_importance[:, start_abs:end_abs] *= 2.0
            
            # E. Continuity Bridging
            active_capsules = [c for c in self.registry.capsules.values() if c.capsule_id not in evicted]
            gaps = self.bridge_engine.identify_gaps(active_capsules)
            for gap in gaps:
                # Protect bridge tokens
                # Similar mapping to abs indices...
                pass

        # Update global offset
        self.global_offset += q_len
        
        # F. Final Pruning
        pruned_pkv, indices = self.geometry.prune_kv(past_key_values, hidden_states=hidden_states)
        return pruned_pkv, indices
