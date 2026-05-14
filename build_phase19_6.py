import os

def create_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)

DECODER_FILES = {
    "confidence_guided_arbitrator.py": '''import torch

class ConfidenceGuidedArbitrator:
    """PHASE 19.6A: Confidence-Guided Decoder Arbitration"""
    def arbitrate_logits(self, logits: torch.Tensor, confidence_score: float, symbolic_bias: float = 2.0) -> torch.Tensor:
        # If we have high symbolic confidence, boost top-K symbolic candidates
        # (Simplified: in real implementation, we'd map confidence to specific token IDs)
        if confidence_score > 0.8:
            logits = logits * symbolic_bias
        return logits
''',
    "symbolic_confidence_router.py": '''class SymbolicConfidenceRouter: pass''',
    "global_identity_biaser.py": '''class GlobalIdentityBiaser: pass''',
    "arbitration_priority_controller.py": '''class ArbitrationPriorityController: pass''',
    
    "sparse_attention_steering.py": '''import torch

class SparseAttentionSteering:
    """PHASE 19.6B: Sparse-Aware Decoder Attention Steering"""
    def steer_attention(self, attention_mask: torch.Tensor, symbolic_indices: torch.Tensor) -> torch.Tensor:
        # Boost attention to symbolic anchors during the decoding step
        if len(symbolic_indices) > 0:
            attention_mask[:, :, :, symbolic_indices] += 5.0
        return attention_mask
''',
    "symbolic_visibility_booster.py": '''class SymbolicVisibilityBooster: pass''',
    "contextual_noise_balancer.py": '''class ContextualNoiseBalancer: pass''',
    "final_step_signal_router.py": '''class FinalStepSignalRouter: pass''',
    
    "consensus_token_selector.py": '''import torch

class ConsensusTokenSelector:
    """PHASE 19.6C: Consensus-Aware Token Selection"""
    def select_tokens(self, probabilities: torch.Tensor, agreement_state: float) -> torch.Tensor:
        # Shift probability mass toward globally agreed continuations
        if agreement_state > 0.9:
            probabilities = torch.pow(probabilities, 0.8) # Sharpen distribution
        return probabilities
''',
    "global_symbolic_state_reader.py": '''class GlobalSymbolicStateReader: pass''',
    "distributed_confidence_sampler.py": '''class DistributedConfidenceSampler: pass''',
    "identity_consistency_validator.py": '''class IdentityConsistencyValidator: pass''',
    
    "symbolic_continuation_tracker.py": '''class SymbolicContinuationTracker:
    def __init__(self):
        self.active_continuation = False
    def update(self, token_id):
        # Track if we are currently mid-symbolic identifier
        pass
''',
    "multi_token_identity_stabilizer.py": '''class MultiTokenIdentityStabilizer: pass''',
    "prefix_suffix_alignment_engine.py": '''class PrefixSuffixAlignmentEngine: pass''',
    "continuation_confidence_monitor.py": '''class ContinuationConfidenceMonitor: pass'''
}

ANALYSIS_FILES = {
    "decoder_overhead_tracker.py": '''import time

class DecoderOverheadTracker:
    def __init__(self):
        self.total_arbitration_time = 0.0
    def record(self, duration):
        self.total_arbitration_time += duration
''',
    "arbitration_efficiency_mapper.py": '''class ArbitrationEfficiencyMapper: pass''',
    "sampling_balance_auditor.py": '''class SamplingBalanceAuditor: pass'''
}

RUNTIME_RESOLVER = '''import torch
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
        self.arbitrator = ConfidenceGuidedArbitrator()
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
        logits = self.arbitrator.arbitrate_logits(logits, self.global_confidence)
        
        # 19.6C: Soft agreement shift
        probs = torch.softmax(logits, dim=-1)
        probs = self.token_selector.select_tokens(probs, self.global_confidence)
        # Convert back to logits (log-probs)
        logits = torch.log(probs + 1e-12)
        
        end.record()
        torch.cuda.synchronize()
        self.decoder_overhead.record(start.elapsed_time(end))
        
        return logits
'''

VALIDATION_SCRIPT = '''import torch
import os
import time
import json
from models.qwen7b_real_loader import Qwen7BRealLoader
from validation.long_context_workload_suite import LongContextWorkloadSuite
from runtime.hierarchical_memory_resolver import HierarchicalMemoryResolver
from runtime.persistent_relevance_resolver import PersistentRelevanceResolver
from runtime.anchor_reinforcement_resolver import AnchorReinforcementResolver
from runtime.hybrid_memory_resolver import HybridMemoryResolver
from runtime.resonance_memory_resolver import ResonanceMemoryResolver
from runtime.discriminative_memory_resolver import DiscriminativeMemoryResolver
from runtime.consensus_memory_resolver import ConsensusMemoryResolver
from runtime.synchronized_memory_resolver import SynchronizedMemoryResolver
from runtime.persistent_memory_resolver import PersistentMemoryResolver
from runtime.guided_memory_resolver import GuidedMemoryResolver
from runtime.adaptive_chunk_overlap import AdaptiveChunkOverlap
from transformers import AutoTokenizer, DynamicCache

class ValidationRunner19_6:
    def __init__(self, model, tokenizer, results_dir="results/reconstruction_19_6/"):
        self.model = model
        self.tokenizer = tokenizer
        self.results_dir = results_dir
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        self.suite = LongContextWorkloadSuite(tokenizer)
        self.overlap_scheduler = AdaptiveChunkOverlap(overlap_size=128)

    def run_matrix(self):
        contexts = [4096, 8192, 16384]
        modes = ["dense", "sparse_baseline", "hmc_18_7", "prmrs_18_8", "arrsbs_18_9", "sbpvcr_19_0", "ssrcrf_19_1", "asdcaf_19_2", "dscrrf_19_3", "hhssgc_19_4", "pgrscs_19_5", "sadacgg_19_6"]
        
        test_id = "SIGMA-19-6-SADACGG-TEST"

        for ctx_len in contexts:
            for mode in modes:
                print(f"\\n[RUN] Mode: {mode} | Context: {ctx_len} | ID: {test_id}")
                result = self.execute_single_run(mode, ctx_len, test_id)
                self.log_result(mode, ctx_len, result)

    def execute_single_run(self, mode, ctx_len, needle_str):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        resolver = None
        if mode == "sparse_baseline":
            resolver = HybridMemoryResolver(anchor_budget=ctx_len // 2)
            resolver.fidelity.threshold = 100.0 
        elif mode == "hmc_18_7":
            resolver = HierarchicalMemoryResolver(anchor_budget=ctx_len // 2, fidelity_token_budget=1024)
        elif mode == "prmrs_18_8":
            resolver = PersistentRelevanceResolver(anchor_budget=ctx_len // 2, fidelity_token_budget=1024)
        elif mode == "arrsbs_18_9":
            resolver = AnchorReinforcementResolver(self.tokenizer, anchor_budget=ctx_len // 2, fidelity_token_budget=1024)
        elif mode == "sbpvcr_19_0":
            resolver = HybridMemoryResolver(anchor_budget=ctx_len // 2, fidelity_budget=1024)
        elif mode == "ssrcrf_19_1":
            resolver = ResonanceMemoryResolver(anchor_budget=ctx_len // 2, fidelity_budget=1024)
        elif mode == "asdcaf_19_2":
            resolver = DiscriminativeMemoryResolver(anchor_budget=ctx_len // 2, fidelity_budget=1024)
        elif mode == "dscrrf_19_3":
            resolver = ConsensusMemoryResolver(anchor_budget=ctx_len // 2, fidelity_budget=1024)
        elif mode == "hhssgc_19_4":
            resolver = SynchronizedMemoryResolver(anchor_budget=ctx_len // 2, fidelity_budget=1024)
        elif mode == "pgrscs_19_5":
            resolver = PersistentMemoryResolver(anchor_budget=ctx_len // 2, fidelity_budget=1024)
        elif mode == "sadacgg_19_6":
            resolver = GuidedMemoryResolver(anchor_budget=ctx_len // 2, fidelity_budget=1024)
        
        test_case = self.suite.create_needle_in_haystack(
            ctx_len, 
            needle=f"The secret activation code for the Differential KV project is {needle_str}. It verifies relational agreement.", 
            answer=needle_str,
            needle_pos_ratio=0.3
        )
        input_ids = torch.tensor([test_case['tokens']]).to("cuda")
        
        past_key_values = DynamicCache()
        chunk_size = 512
        chunks = self.overlap_scheduler.get_chunks(input_ids, chunk_size)
        
        start_time = time.perf_counter()
        
        for chunk, _, _ in chunks:
            with torch.no_grad():
                outputs = self.model(input_ids=chunk, past_key_values=past_key_values, use_cache=True, output_hidden_states=True)
            
            if mode != "dense":
                legacy = past_key_values.to_legacy_cache()
                pruned, meta = resolver.resolve_and_prune(legacy, outputs.hidden_states[-1], chunk)
                past_key_values = DynamicCache.from_legacy_cache(pruned)
        
        prefill_end = time.perf_counter()
        ttft = prefill_end - start_time
        
        curr_input = input_ids[:, -1:]
        response_tokens = []
        generated_text = ""
        
        gen_start = time.perf_counter()
        for _ in range(64):
            with torch.no_grad():
                outputs = self.model(input_ids=curr_input, past_key_values=past_key_values, use_cache=True)
                logits = outputs.logits[:, -1, :] / 0.7
                
                # 19.6: Guidance injection
                if mode == "sadacgg_19_6":
                    logits = resolver.guide_decoder(logits)
                
                token = torch.argmax(logits, dim=-1).unsqueeze(0)
                
                response_tokens.append(token.item())
                curr_input = token
                new_text = self.tokenizer.decode([token.item()])
                generated_text += new_text
                if self.tokenizer.eos_token in new_text: break
        
        gen_end = time.perf_counter()
        tps = len(response_tokens) / (gen_end - gen_start) if (gen_end - gen_start) > 0 else 0
        vram = torch.cuda.max_memory_allocated() / (1024**3)
        
        exact_match = needle_str in generated_text
        prefix_match = needle_str[:8] in generated_text
        
        return {
            "success": exact_match,
            "prefix_match": prefix_match,
            "tps": tps,
            "ttft": ttft,
            "vram_gb": vram,
            "output": generated_text,
            "expected": needle_str,
            "arbitration_overhead": getattr(resolver, 'decoder_overhead', None).total_arbitration_time if hasattr(resolver, 'decoder_overhead') and resolver.decoder_overhead else 0.0,
            "confidence": getattr(resolver, 'global_confidence', 0.0) if hasattr(resolver, 'global_confidence') else 0.0
        }

    def log_result(self, mode, ctx, res):
        log_file = os.path.join(self.results_dir, "raw_decoder_arbitration.jsonl")
        with open(log_file, "a") as f:
            f.write(json.dumps({"mode": mode, "ctx": ctx, **res}) + "\\n")
        print(f"  [RESULT] EM: {res['success']} | Prefix: {res['prefix_match']} | TPS: {res['tps']:.2f} | VRAM: {res['vram_gb']:.2f}GB")

def main():
    print("="*60)
    print("PHASE 19.6 - SPARSE-AWARE DECODER ARBITRATION")
    print("="*60)
    
    loader = Qwen7BRealLoader()
    model = loader.load()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    
    runner = ValidationRunner19_6(model, tokenizer)
    runner.run_matrix()

if __name__ == "__main__":
    main()
'''

def run():
    for f_name, content in DECODER_FILES.items():
        create_file(os.path.join("d:/Codes/Projects/Differential KV/decoder", f_name), content)
    for f_name, content in ANALYSIS_FILES.items():
        create_file(os.path.join("d:/Codes/Projects/Differential KV/analysis", f_name), content)
    create_file("d:/Codes/Projects/Differential KV/runtime/guided_memory_resolver.py", RUNTIME_RESOLVER)
    create_file("d:/Codes/Projects/Differential KV/run_reconstruction_19_6_validation.py", VALIDATION_SCRIPT)

if __name__ == "__main__":
    run()
