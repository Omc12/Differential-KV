
import torch
from typing import Optional, List, Dict
from runtime.spslrif_resolver import SPSLRIFResolver
from memory.attention_mass_profiler import AttentionMassProfiler
from memory.structural_anchor_booster import StructuralAnchorBooster, DelimiterIntegrityField
from memory.hub_anchor_registry import HubAnchorRegistry
from memory.attention_energy_compressor import AttentionEnergyCompressor
from memory.symbolic_focus_router import SymbolicFocusRouter

class SABEAFResolver(SPSLRIFResolver):
    """
    PHASE 20.8: SABEAF (Structural Anchor Boosting & Attention Energy Focusing).
    Targets 'Attention Mass Dilution' at 16k context.
    """
    def __init__(self, tokenizer, anchor_budget: int = 6144, fidelity_budget: int = 1024):
        super().__init__(tokenizer, anchor_budget, fidelity_budget)
        self.profiler = AttentionMassProfiler()
        self.anchor_booster = StructuralAnchorBooster(tokenizer)
        self.integrity_field = DelimiterIntegrityField(self.anchor_booster)
        self.hub_registry = HubAnchorRegistry(tokenizer)
        self.compressor = AttentionEnergyCompressor()
        self.focus_router = SymbolicFocusRouter()
        
        # Telemetry
        self.attention_density_log = []
        self.hub_retrieval_log = []
        self.drift_risk = 0.0

    def resolve_and_prune(self, past_key_values, hidden_states, chunk_input_ids, attention_probs=None):
        """Standard resolve/prune + Hub Registration."""
        # 1. Base logic (Booster population)
        pruned_pkv, indices = super().resolve_and_prune(past_key_values, hidden_states, chunk_input_ids, attention_probs)
        
        # 2. Hub Registration (20.8)
        # Check if new spans were added to the booster and register them as hubs
        # We use the global offset to identify the origin
        if len(self.booster.active_spans) > 0:
            start, end = self.booster.active_spans[-1]
            if start >= self.global_offset - chunk_input_ids.shape[1]:
                if len(self.booster.ordered_sequences) > 0:
                    last_tokens = self.booster.ordered_sequences[-1]
                    self.hub_registry.register_root(start, last_tokens)
                
        return pruned_pkv, indices

    def guide_decoder(self, logits: torch.Tensor, attention_weights: torch.Tensor = None) -> torch.Tensor:
        """
        PHASE 20.8: SABEAF - Focused Attention Energy Amplification.
        """
        # 1. Base SPSLRIF Logic (Lineage + Momentum)
        calibrated_logits = super().guide_decoder(logits, attention_weights)
        
        # 2. Attention Mass Profiling & Compression (20.8)
        drift_risk = 0.0
        if attention_weights is not None:
            # Calculate current symbolic mask for the profiler
            current_abs_indices = self.geometry.absolute_indices[0]
            mass = attention_weights[-1, 0].mean(dim=0)[-1]
            steering_mask = self.booster.get_steering_bias(current_abs_indices, mass.device)
            # Fix: mass and steering_mask must have same length
            min_len = min(mass.shape[0], steering_mask.shape[0])
            mass = mass[:min_len]
            steering_mask = steering_mask[:min_len]
                
            # Energy Compression
            compressed_mass = self.compressor.compress_neighborhoods(mass, steering_mask)
            
            profile = self.profiler.profile_attention(compressed_mass, steering_mask)
            self.attention_density_log.append(profile)
            
            # 3. Energy Focusing & Routing
            self.drift_risk = self.focus_router.calculate_drift_risk(
                profile["fragmentation"], self.integrity_field.drift_detected
            )
            
            # Route focus toward locked tokens if risk is high
            locked_ids = self.booster.get_locked_token_ids()
            if locked_ids:
                calibrated_logits = self.focus_router.route_focus(
                    calibrated_logits, locked_ids, self.drift_risk
                )
                
            # If fragmentation is very high, apply extra focus amplification (local)
            if profile["fragmentation"] > 6.0:
                amp = self.compressor.get_focus_multiplier(profile["fragmentation"])
                if locked_ids:
                    t_idx = torch.tensor(locked_ids, device=calibrated_logits.device)
                    t_idx = t_idx[t_idx < calibrated_logits.shape[-1]]
                    calibrated_logits[0, t_idx] *= amp

        # 4. Structural Anchor Boosting (20.8)
        # Identify if we are expecting a delimiter and boost it
        expected_token = self._get_current_expected_anchor()
        if expected_token is not None:
            is_delimiter = expected_token in self.anchor_booster.delimiter_ids
            amp_factor = self.integrity_field.get_amplification_factor()
            
            boost_val = self.anchor_booster.delimiter_boost if is_delimiter else self.anchor_booster.base_boost
            calibrated_logits[0, expected_token] += boost_val * amp_factor * (1.0 + self.drift_risk)

        # 5. Hub-Assisted Retrieval (Experimental)
        # If the model seems to be "searching" for a hub (high focus on a root)
        # We can bias toward the first token of that hub if we are at a pivot point
        # For now, this is purely probabilistic via the focus router's routing.

        return calibrated_logits

    def record_generated_token(self, token_id: int, logits: torch.Tensor):
        """Phase 20.8: Delimiter Integrity Tracking."""
        expected_token = self._get_current_expected_anchor()
        if expected_token is not None:
            is_delimiter = expected_token in self.anchor_booster.delimiter_ids
            if is_delimiter:
                is_match = (token_id == expected_token)
                self.integrity_field.update(token_id, is_match)
                
        # Call parent for momentum/lineage updates
        super().record_generated_token(token_id, logits)
