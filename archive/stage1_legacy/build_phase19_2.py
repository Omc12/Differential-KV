import os

def create_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)

MEMORY_FILES = {
    "adaptive_signal_discriminator.py": '''import torch

class AdaptiveSignalDiscriminator:
    """PHASE 19.2A: Adaptive Signal Discrimination"""
    def discriminate(self, scores: torch.Tensor, identity_mask: torch.Tensor) -> torch.Tensor:
        # Boost signals that have high identity uniqueness
        boost = torch.zeros_like(scores)
        boost[identity_mask] = 2000.0
        return scores + boost
''',
    "symbolic_identity_tracker.py": '''import torch

class SymbolicIdentityTracker:
    def track_identity(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Tracks uniqueness signature
        return torch.abs(hidden_states).mean(dim=-1) > 1.5
''',
    "contrastive_signal_mapper.py": '''class ContrastiveSignalMapper: pass''',
    "discriminative_pathway_analyzer.py": '''class DiscriminativePathwayAnalyzer: pass''',
    
    "contrastive_attention_field.py": '''import torch

class ContrastiveAttentionField:
    """PHASE 19.2B: Contrastive Attention Field"""
    def apply_contrast(self, scores: torch.Tensor, noise_mask: torch.Tensor) -> torch.Tensor:
        # Suppress noise locally
        scores[noise_mask] *= 0.1
        return scores
''',
    "local_noise_suppressor.py": '''import torch

class LocalNoiseSuppressor:
    def detect_noise(self, scores: torch.Tensor, symbolic_mask: torch.Tensor) -> torch.Tensor:
        # High scores that are not symbolic are considered noise
        threshold = scores.mean() * 2.0
        return (scores > threshold) & (~symbolic_mask)
''',
    "signal_visibility_allocator.py": '''class SignalVisibilityAllocator: pass''',
    "attention_competition_controller.py": '''class AttentionCompetitionController: pass''',
    
    "symbolic_identity_reinforcer.py": '''class SymbolicIdentityReinforcer: pass''',
    "uniqueness_signature_tracker.py": '''class UniquenessSignatureTracker: pass''',
    "identity_resonance_mapper.py": '''class IdentityResonanceMapper: pass''',
    "contextual_identity_stabilizer.py": '''class ContextualIdentityStabilizer: pass''',
    
    "adaptive_attention_prioritizer.py": '''class AdaptiveAttentionPrioritizer: pass''',
    "pathway_visibility_scheduler.py": '''class PathwayVisibilityScheduler: pass''',
    "traversal_importance_allocator.py": '''class TraversalImportanceAllocator: pass''',
    "sparse_signal_router.py": '''class SparseSignalRouter: pass'''
}

ANALYSIS_FILES = {
    "snr_efficiency_tracker.py": '''class SNREfficiencyTracker:
    def __init__(self):
        self.snr_gain = 0.0
    def record_gain(self, gain: float):
        self.snr_gain += gain
''',
    "contrastive_overhead_mapper.py": '''class ContrastiveOverheadMapper: pass''',
    "attention_visibility_auditor.py": '''class AttentionVisibilityAuditor: pass'''
}

RUNTIME_RESOLVER = '''import torch
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
            noise_mask = self.noise_suppressor.detect_noise(self.geometry.accumulated_importance, symbolic_mask) # symbolic_mask is for current chunk
            # Simplified: suppress noise in the whole accumulated buffer
            # Actually we only have symbolic_mask for the current chunk here.
            # We'd need a global symbolic mask to do it for the whole buffer properly.
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
from runtime.adaptive_chunk_overlap import AdaptiveChunkOverlap
from transformers import AutoTokenizer, DynamicCache

class ValidationRunner19_2:
    def __init__(self, model, tokenizer, results_dir="results/reconstruction_19_2/"):
        self.model = model
        self.tokenizer = tokenizer
        self.results_dir = results_dir
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        self.suite = LongContextWorkloadSuite(tokenizer)
        self.overlap_scheduler = AdaptiveChunkOverlap(overlap_size=128)

    def run_matrix(self):
        contexts = [4096, 8192, 16384]
        modes = ["dense", "sparse_baseline", "hmc_18_7", "prmrs_18_8", "arrsbs_18_9", "sbpvcr_19_0", "ssrcrf_19_1", "asdcaf_19_2"]
        
        test_id = "SIGMA-19-2-ASDCAF-TEST"

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
        
        test_case = self.suite.create_needle_in_haystack(
            ctx_len, 
            needle=f"The discriminative signature for Phase 19.2 is {needle_str}. Use it for contrastive validation.", 
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
            "expected": needle_str
        }

    def log_result(self, mode, ctx, res):
        log_file = os.path.join(self.results_dir, "raw_signal_discrimination.jsonl")
        with open(log_file, "a") as f:
            f.write(json.dumps({"mode": mode, "ctx": ctx, **res}) + "\\n")
        print(f"  [RESULT] EM: {res['success']} | Prefix: {res['prefix_match']} | TPS: {res['tps']:.2f} | VRAM: {res['vram_gb']:.2f}GB")

def main():
    print("="*60)
    print("PHASE 19.2 - ADAPTIVE SIGNAL DISCRIMINATION & CONTRASTIVE ATTENTION")
    print("="*60)
    
    loader = Qwen7BRealLoader()
    model = loader.load()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    
    runner = ValidationRunner19_2(model, tokenizer)
    runner.run_matrix()

if __name__ == "__main__":
    main()
'''

def run():
    for f_name, content in MEMORY_FILES.items():
        create_file(os.path.join("d:/Codes/Projects/Differential KV/memory", f_name), content)
    for f_name, content in ANALYSIS_FILES.items():
        create_file(os.path.join("d:/Codes/Projects/Differential KV/analysis", f_name), content)
    create_file("d:/Codes/Projects/Differential KV/runtime/discriminative_memory_resolver.py", RUNTIME_RESOLVER)
    create_file("d:/Codes/Projects/Differential KV/run_reconstruction_19_2_validation.py", VALIDATION_SCRIPT)

if __name__ == "__main__":
    run()
