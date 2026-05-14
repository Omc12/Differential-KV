import torch
from runtime.resonance_memory_resolver import ResonanceMemoryResolver
from memory.adaptive_signal_discriminator import AdaptiveSignalDiscriminator
from memory.contrastive_attention_field import ContrastiveAttentionField
from memory.symbolic_identity_tracker import SymbolicIdentityTracker
from memory.local_noise_suppressor import LocalNoiseSuppressor
from analysis.snr_efficiency_tracker import SNREfficiencyTracker

class DiscriminativeMemoryResolver(ResonanceMemoryResolver):
    """PHASE 19.2: ASDCAF Resolver"""
    def __init__(self, anchor_budget: int = 6144, fidelity_budget: int = 512):
        super().__init__(anchor_budget, fidelity_budget)
        self.discriminator = AdaptiveSignalDiscriminator()
        self.contrast_field = ContrastiveAttentionField()
        self.identity_tracker = SymbolicIdentityTracker()
        self.noise_suppressor = LocalNoiseSuppressor()
        self.snr_tracker = SNREfficiencyTracker()

    def resolve_and_prune(self, past_key_values, hidden_states, chunk_input_ids, attention_probs=None):
        self.overhead_tracker.start_measure()
        
        symbolic_mask = self.fidelity.detect_high_entropy_tokens(hidden_states)
        q_len = hidden_states.shape[1]
        
        chunk_indices = torch.arange(self.global_offset, self.global_offset + q_len, device=hidden_states.device)
        self.global_offset += q_len
        
        all_sym_indices = self.fidelity.fidelity_indices if self.fidelity.fidelity_indices is not None else torch.tensor([], dtype=torch.long, device=hidden_states.device)
        current_sym_idx = chunk_indices[symbolic_mask[0]]
        all_sym_indices = torch.unique(torch.cat([all_sym_indices, current_sym_idx]))
        
        if self.geometry.accumulated_importance is not None:
            # 19.0 & 19.1 steps
            self.geometry.accumulated_importance = self.smoother.smooth_importance(self.geometry.accumulated_importance)
            self.geometry.accumulated_importance = self.slope_preserver.preserve_slopes(
                self.geometry.accumulated_importance, self.geometry.absolute_indices, all_sym_indices)
            self.geometry.accumulated_importance = self.runway.apply_runway(self.geometry.accumulated_importance, current_sym_idx)
            
            energy = self.energy_tracker.track_energy(hidden_states)
            decay = self.decay_compensator.calculate_decay(energy)
            if decay.shape[1] == self.geometry.accumulated_importance.shape[1]:
                self.geometry.accumulated_importance = self.resonance_hubs.activate_hubs(self.geometry.accumulated_importance, decay)
                self.cost_tracker.add_activation()
                
            # Phase 19.2 ASDCAF logic
            # A. Signal Discrimination
            identity_mask = self.identity_tracker.track_identity(hidden_states)
            # Map chunk identity mask to global accumulated importance if possible
            # Simplified: just use current chunk region in accumulated importance
            if identity_mask.shape[1] <= self.geometry.accumulated_importance.shape[1]:
                # We need to know which absolute indices in accumulated_importance correspond to the CURRENT chunk
                # Since prune_kv hasn't been called yet, the tail of accumulated_importance IS the current chunk.
                self.geometry.accumulated_importance[:, -q_len:] = self.discriminator.discriminate(
                    self.geometry.accumulated_importance[:, -q_len:], identity_mask)
            
            # B. Contrastive Attention Fields
            # For now, apply contrast to the current chunk tail
            noise_mask_chunk = self.noise_suppressor.detect_noise(self.geometry.accumulated_importance[:, -q_len:], symbolic_mask)
            self.geometry.accumulated_importance[:, -q_len:] = self.contrast_field.apply_contrast(
                self.geometry.accumulated_importance[:, -q_len:], noise_mask_chunk)

        bridge_mask = self.bridge_router.route_bridges(
            self.geometry.accumulated_importance if self.geometry.accumulated_importance is not None else torch.zeros((1, 1), device=hidden_states.device), 
            self.geometry.absolute_indices,
            all_sym_indices)
        
        if self.geometry.accumulated_importance is not None and bridge_mask.shape[1] == self.geometry.accumulated_importance.shape[1]:
            dtype_max = torch.finfo(self.geometry.accumulated_importance.dtype).max
            self.geometry.accumulated_importance[bridge_mask] = dtype_max

        pruned_pkv, indices = self.geometry.prune_kv(past_key_values, hidden_states=hidden_states)
        
        num_bridges = bridge_mask.sum().item()
        self.overhead_tracker.end_measure(int(num_bridges))
        
        return pruned_pkv, indices
