import torch
from runtime.synchronized_memory_resolver import SynchronizedMemoryResolver
from memory.global_reanchoring_pulses import GlobalReanchoringPulses
from memory.persistent_symbolic_state import PersistentSymbolicState
from memory.hierarchical_certainty_heartbeats import HierarchicalCertaintyHeartbeats
from memory.certainty_drift_tracker import CertaintyDriftTracker
from memory.adaptive_certainty_recovery import AdaptiveCertaintyRecovery
from analysis.reanchoring_cost_tracker import ReanchoringCostTracker

class PersistentMemoryResolver(SynchronizedMemoryResolver):
    """PHASE 19.5: PGRSCS Resolver"""
    def __init__(self, anchor_budget: int = 6144, fidelity_budget: int = 512):
        super().__init__(anchor_budget, fidelity_budget)
        self.pulses = GlobalReanchoringPulses()
        self.persistent_state = PersistentSymbolicState()
        self.heartbeats = HierarchicalCertaintyHeartbeats()
        self.drift_tracker = CertaintyDriftTracker()
        self.recovery = AdaptiveCertaintyRecovery()
        self.reanchor_tracker = ReanchoringCostTracker()
        self.step_count = 0

    def resolve_and_prune(self, past_key_values, hidden_states, chunk_input_ids, attention_probs=None):
        self.overhead_tracker.start_measure()
        self.step_count += 1
        
        symbolic_mask = self.fidelity.detect_high_entropy_tokens(hidden_states)
        q_len = hidden_states.shape[1]
        
        chunk_indices = torch.arange(self.global_offset, self.global_offset + q_len, device=hidden_states.device)
        self.global_offset += q_len
        
        all_sym_indices = self.fidelity.fidelity_indices if self.fidelity.fidelity_indices is not None else torch.tensor([], dtype=torch.long, device=hidden_states.device)
        current_sym_idx = chunk_indices[symbolic_mask[0]]
        all_sym_indices = torch.unique(torch.cat([all_sym_indices, current_sym_idx]))
        
        if self.geometry.accumulated_importance is not None:
            # Previous phase steps
            self.geometry.accumulated_importance = self.smoother.smooth_importance(self.geometry.accumulated_importance)
            self.geometry.accumulated_importance = self.slope_preserver.preserve_slopes(
                self.geometry.accumulated_importance, self.geometry.absolute_indices, all_sym_indices)
            self.geometry.accumulated_importance = self.runway.apply_runway(self.geometry.accumulated_importance, current_sym_idx)
            
            energy = self.energy_tracker.track_energy(hidden_states)
            decay = self.decay_compensator.calculate_decay(energy)
            if decay.shape[1] == self.geometry.accumulated_importance.shape[1]:
                self.geometry.accumulated_importance = self.resonance_hubs.activate_hubs(self.geometry.accumulated_importance, decay)
            
            identity_mask = self.identity_tracker.track_identity(hidden_states)
            if identity_mask.shape[1] <= self.geometry.accumulated_importance.shape[1]:
                self.geometry.accumulated_importance[:, -q_len:] = self.discriminator.discriminate(
                    self.geometry.accumulated_importance[:, -q_len:], identity_mask)
            
            noise_mask_chunk = self.noise_suppressor.detect_noise(self.geometry.accumulated_importance[:, -q_len:], symbolic_mask)
            self.geometry.accumulated_importance[:, -q_len:] = self.contrast_field.apply_contrast(
                self.geometry.accumulated_importance[:, -q_len:], noise_mask_chunk)
            
            multipath_evidence = self.multipath_tracker.track_multipath(hidden_states)
            if multipath_evidence.shape[1] <= self.geometry.accumulated_importance.shape[1]:
                self.geometry.accumulated_importance[:, -q_len:] = self.consensus.accumulate_votes(
                    self.geometry.accumulated_importance[:, -q_len:], multipath_evidence)

            consensus_score = (self.geometry.accumulated_importance > 10000.0).float().mean().item()
            self.geometry.accumulated_importance = self.hub_network.synchronize_hubs(
                self.geometry.accumulated_importance, consensus_score)
            self.sync_tracker.add_sync()

            if len(all_sym_indices) > 0:
                current_abs_indices = self.geometry.absolute_indices[0]
                matches = []
                for sym_idx in all_sym_indices[-10:].tolist():
                    m = (current_abs_indices == int(sym_idx)).nonzero()
                    if len(m) > 0: matches.append(m[0].item())
                
                if len(matches) > 0:
                    relay_indices = torch.tensor(matches[:5], device=hidden_states.device)
                    beacon_indices = torch.tensor(matches[5:], device=hidden_states.device)
                    
                    if len(relay_indices) > 0:
                        self.geometry.accumulated_importance = self.relays.propagate_signal(
                            self.geometry.accumulated_importance, relay_indices)
                    if len(beacon_indices) > 0:
                        self.geometry.accumulated_importance = self.beacons.broadcast_beacons(
                            self.geometry.accumulated_importance, beacon_indices)

            self.geometry.accumulated_importance = self.arbitrator.arbitrate(self.geometry.accumulated_importance)

            # Phase 19.5 PGRSCS logic
            # A. Global Re-anchoring Pulse
            if self.step_count % 4 == 0:
                # Find indices of all historical symbolic targets in the current cache
                current_abs_indices = self.geometry.absolute_indices[0]
                # Vectorized search for speed
                mask = torch.isin(current_abs_indices, all_sym_indices)
                sym_cache_indices = mask.nonzero().squeeze(-1)
                
                if len(sym_cache_indices) > 0:
                    self.geometry.accumulated_importance = self.pulses.apply_pulse(
                        self.geometry.accumulated_importance, sym_cache_indices)
                    self.reanchor_tracker.add_pulse()

            # B. Heartbeats & Drift Recovery
            self.geometry.accumulated_importance = self.heartbeats.heartbeat(
                self.geometry.accumulated_importance, self.step_count)
            
            self.drift_tracker.track_drift(self.geometry.accumulated_importance)
            self.geometry.accumulated_importance = self.recovery.recover(
                self.geometry.accumulated_importance, self.drift_tracker.drift_score)

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
