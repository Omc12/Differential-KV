
import torch
import time
from typing import Optional, List, Dict, Any, Tuple
from runtime.hsha_resolver import HSHAResolver
from hsha import (
    RecallLegitimacyScorer,
    FalseRecallSuppressor,
    EntropyCompatibilityGate,
    MulticandidateRecallRouter,
    RecallDecayController
)

class SRLResolver(HSHAResolver):
    """
    PHASE 21.1: SRL (Symbolic Recall Legitimacy).
    Builds legitimacy-aware symbolic recall systems for Hybrid Symbolic Hub Architecture.
    Targets 'False Recall Suppression' and 'Entropy Preservation'.
    """
    def __init__(self, tokenizer, anchor_budget: int = 6144, fidelity_budget: int = 1024):
        super().__init__(tokenizer, anchor_budget, fidelity_budget)
        
        # SRL Core Modules
        self.legitimacy_scorer = RecallLegitimacyScorer(tokenizer)
        self.suppressor = FalseRecallSuppressor()
        self.entropy_gate = EntropyCompatibilityGate()
        self.multi_router = MulticandidateRecallRouter(self.legitimacy_scorer)
        self.recall_decay_controller = RecallDecayController()
        
        # SRL Metrics
        self.srl_stats = {
            "false_recall_events": 0,
            "suppressed_candidates": 0,
            "mean_legitimacy": 0.0,
            "mean_reinjection_strength": 0.0,
            "entropy_preservation_ratio": 1.0,
            "routing_latency": []
        }

    def resolve_and_prune(self, past_key_values, hidden_states, chunk_input_ids, attention_probs=None):
        """
        Extends HSHA resolve_and_prune to sync the SRL suppressor.
        """
        pruned_pkv, indices = super().resolve_and_prune(past_key_values, hidden_states, chunk_input_ids, attention_probs)
        
        # Ensure the suppressor knows about newly active hubs so they aren't marked as 'stale' immediately
        for hub_id in self.active_hubs:
            if hub_id not in self.suppressor.active_hubs_last_pos:
                print(f"[DEBUG] SRL: Syncing suppressor for hub {hub_id} at global pos {self.global_offset}")
                self.suppressor.update_access(hub_id, self.global_offset)
                
        return pruned_pkv, indices

    def guide_decoder(self, logits: torch.Tensor, attention_weights: torch.Tensor = None) -> torch.Tensor:
        """
        PHASE 21.1: SRL - Legitimacy-Aware Symbolic Guidance.
        Ensures recall is contextual and probabilistic, not brute-force.
        """
        # 1. Base Guidance (SABEAF Anchor Boosting + Energy Focusing)
        # We use super(HSHAResolver, self) to skip HSHA's naive recall logic
        from runtime.sabeaf_resolver import SABEAFResolver
        calibrated_logits = super(HSHAResolver, self).guide_decoder(logits, attention_weights)
        
        start_time = time.time()
        
        # 2. SRL Candidate Filtering (Suppress stale/hallucinated hubs)
        # Current position is estimated via global_offset
        candidates = self.suppressor.filter_candidates(
            self.active_hubs, self.global_offset, self.context_tokens
        )
        self.srl_stats["suppressed_candidates"] += (len(self.active_hubs) - len(candidates))
        
        # 3. Multicandidate Routing (Weighted legitimacy)
        scored_candidates = self.multi_router.route_multi(
            self.context_tokens, candidates, self.hub_registry
        )
        
        if scored_candidates:
            hub_id, legitimacy = scored_candidates[0]
            hub_obj = self.hub_registry.get_object(hub_id)
            print(f"[DEBUG] SRL: Scored {len(scored_candidates)} candidates. Best: {hub_id} score={legitimacy:.3f}")
            
            # 4. Synchronize Position in Hub
            # (Improved alignment for multi-candidate support)
            best_idx = 0
            for i in range(len(hub_obj.tokens) - 3):
                if self.context_tokens[-4:] == hub_obj.tokens[i:i+4]:
                    best_idx = i + 4
                    break
            
            if best_idx < len(hub_obj.tokens):
                target_token = hub_obj.tokens[best_idx]
                
                # 5. False Recall Suppression (Legitimacy Check)
                if self.suppressor.detect_hallucination(calibrated_logits, target_token):
                    self.srl_stats["false_recall_events"] += 1
                    # Log the suppressed recall event
                else:
                    # 6. Entropy-Aware Injection
                    entropy = self.entropy_gate.calculate_entropy(calibrated_logits)
                    strength = self.legitimacy_scorer.calculate_injection_strength(legitimacy, entropy)
                    
                    # Apply Decay & Fixation Prevention
                    self.recall_decay_controller.update_reinforcement(hub_id, strength)
                    decayed_strength = self.recall_decay_controller.apply_decay(hub_id)
                    
                    if decayed_strength > 0:
                        # Soft Blending via Entropy Gate
                        print(f"[DEBUG] SRL: Injecting {hub_id} (target_token={target_token}) strength={decayed_strength:.3f}")
                        calibrated_logits = self.entropy_gate.blend_softly(
                            calibrated_logits, target_token, decayed_strength
                        )
                        
                        # Update Metrics
                        self.srl_stats["mean_legitimacy"] = (self.srl_stats["mean_legitimacy"] * 0.9) + (legitimacy * 0.1)
                        self.srl_stats["mean_reinjection_strength"] = (self.srl_stats["mean_reinjection_strength"] * 0.9) + (decayed_strength * 0.1)
                        self.suppressor.update_access(hub_id, self.global_offset)
                        
                        # Track the active recall state for integrity updates
                        self.current_hub_id = hub_id
                        self.hub_token_idx = best_idx

        self.srl_stats["routing_latency"].append(time.time() - start_time)
        return calibrated_logits

    def get_srl_summary(self) -> Dict[str, Any]:
        """Returns summarized metrics for SRL validation."""
        hsha_stats = super().get_hsha_stats()
        
        latency = sum(self.srl_stats["routing_latency"]) / len(self.srl_stats["routing_latency"]) if self.srl_stats["routing_latency"] else 0
        
        return {
            **hsha_stats,
            "false_recall_rate": self.srl_stats["false_recall_events"] / (hsha_stats["hub_utilization"] + 1),
            "mean_legitimacy": self.srl_stats["mean_legitimacy"],
            "reinjection_strength": self.srl_stats["mean_reinjection_strength"],
            "recall_latency_ms": latency * 1000,
            "suppressed_candidates": self.srl_stats["suppressed_candidates"],
            "entropy_preservation": self.entropy_gate.last_entropy
        }
