import torch
from runtime.persistent_memory_resolver import PersistentMemoryResolver
from decoder.confidence_guided_arbitrator import ConfidenceGuidedArbitrator
from decoder.sparse_attention_steering import SparseAttentionSteering
from decoder.consensus_token_selector import ConsensusTokenSelector
from decoder.symbolic_continuation_tracker import SymbolicContinuationTracker
from analysis.decoder_overhead_tracker import DecoderOverheadTracker

class GuidedMemoryResolver(PersistentMemoryResolver):
    """PHASE 19.6: SADACGG Resolver"""
    def __init__(self, anchor_budget: int = 6144, fidelity_budget: int = 512):
        super().__init__(anchor_budget, fidelity_budget)
        self.decoder_arbitrator = ConfidenceGuidedArbitrator()
        self.steering = SparseAttentionSteering()
        self.token_selector = ConsensusTokenSelector()
        self.continuation = SymbolicContinuationTracker()
        self.decoder_overhead = DecoderOverheadTracker()
        self.global_confidence = 0.0

    def resolve_and_prune(self, past_key_values, hidden_states, chunk_input_ids, attention_probs=None):
        # Re-use Phase 19.5 logic
        res = super().resolve_and_prune(past_key_values, hidden_states, chunk_input_ids, attention_probs)
        
        # Update global confidence for decoder guidance
        if self.geometry.accumulated_importance is not None:
            self.global_confidence = (self.geometry.accumulated_importance > 10000.0).float().mean().item()
        
        return res

    def guide_decoder(self, logits: torch.Tensor) -> torch.Tensor:
        """Bias logits based on current symbolic certainty"""
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        
        # 19.6A: Soft guidance
        logits = self.decoder_arbitrator.arbitrate_logits(logits, self.global_confidence)
        
        # 19.6C: Soft agreement shift
        probs = torch.softmax(logits, dim=-1)
        probs = self.token_selector.select_tokens(probs, self.global_confidence)
        # Convert back to logits (log-probs)
        logits = torch.log(probs + 1e-12)
        
        end.record()
        torch.cuda.synchronize()
        self.decoder_overhead.record(start.elapsed_time(end))
        
        return logits
