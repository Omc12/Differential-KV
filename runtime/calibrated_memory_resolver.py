import torch
from runtime.guided_memory_resolver import GuidedMemoryResolver
from decoder.decoder_trust_calibrator import DecoderTrustCalibrator
from decoder.confidence_alignment_engine import ConfidenceAlignmentEngine
from decoder.symbolic_trust_scheduler import SymbolicTrustScheduler
from decoder.probabilistic_identity_biaser import ProbabilisticIdentityBiaser
from decoder.contextual_competition_suppressor import ContextualCompetitionSuppressor
from decoder.contextual_balancing import NoiseMassBalancer
from decoder.symbolic_propagation import SymbolicConfidencePropagator
from decoder.probabilistic_feedback import DynamicSamplingCorrector
from analysis.trust_balance import TrustAlignmentOverhead

class CalibratedMemoryResolver(GuidedMemoryResolver):
    """PHASE 19.7: DTASCC Resolver"""
    def __init__(self, anchor_budget: int = 6144, fidelity_budget: int = 512):
        super().__init__(anchor_budget, fidelity_budget)
        self.trust_calibrator = DecoderTrustCalibrator()
        self.alignment_engine = ConfidenceAlignmentEngine()
        self.trust_scheduler = SymbolicTrustScheduler()
        self.identity_biaser = ProbabilisticIdentityBiaser()
        self.competition_suppressor = ContextualCompetitionSuppressor()
        self.noise_balancer = NoiseMassBalancer()
        self.confidence_propagator = SymbolicConfidencePropagator()
        self.sampling_corrector = DynamicSamplingCorrector()
        self.trust_overhead = TrustAlignmentOverhead()
        
        self.current_trust = 1.0

    def guide_decoder(self, logits: torch.Tensor) -> torch.Tensor:
        """PHASE 19.7: Calibrated Decoder Trust Alignment"""
        # Ensure input logits are not NaN
        if torch.isnan(logits).any():
            logits = torch.nan_to_num(logits)
            
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        
        # 1. Calibrate Trust based on global confidence + momentum
        raw_conf = getattr(self, 'global_confidence', 0.5)
        calibrated_conf = self.confidence_propagator.propagate(raw_conf)
        
        # Ensure confidence is within [0, 1]
        calibrated_conf = max(0.0, min(1.0, float(calibrated_conf)))
        
        self.current_trust = self.trust_calibrator.calibrate(calibrated_conf)
        self.current_trust = self.alignment_engine.align_trust(self.current_trust, calibrated_conf)
        self.current_trust = self.trust_scheduler.get_intensity(self.current_trust)
        
        # Clamp trust to safe levels
        self.current_trust = max(1.0, min(10.0, float(self.current_trust)))
        
        # 2. Apply Probabilistic Identity Bias (Shift toward signal)
        logits = self.identity_biaser.apply_bias(logits, self.current_trust)
        
        # 3. Suppress Contextual Competition
        # Use a safe softmax
        probs = torch.softmax(logits.float(), dim=-1)
        probs = self.competition_suppressor.suppress(probs, calibrated_conf)
        probs = self.noise_balancer.balance(probs, calibrated_conf)
        
        # 4. Sharpen Sampling if needed
        temp_factor = self.sampling_corrector.correct_temperature(1.0, self.current_trust)
        # Ensure temp_factor is safe (not too small)
        temp_factor = max(0.05, float(temp_factor))
        
        if temp_factor < 1.0:
            # Safe power operation
            probs = torch.pow(probs, 1.0 / temp_factor)
            denom = probs.sum(dim=-1, keepdim=True)
            if denom > 1e-12:
                probs = probs / denom
            else:
                # Fallback to original distribution if everything collapsed
                probs = torch.softmax(logits.float(), dim=-1)
            
        # Safe log
        logits = torch.log(probs + 1e-12)
        
        end.record()
        torch.cuda.synchronize()
        self.trust_overhead.record(start.elapsed_time(end))
        
        return logits.to(torch.float16)
        
    def record_generated_token(self, token_id: int, logits: torch.Tensor):
        """Phase 19.7+: Placeholder for generation-time tracking."""
        pass
