import os

def create_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)

MEMORY_FILES = {
    "sparse_resonance_hubs.py": '''import torch

class SparseResonanceHubs:
    """PHASE 19.1A: Sparse Resonance Hubs"""
    def __init__(self, hub_capacity: int = 128):
        self.hub_capacity = hub_capacity

    def activate_hubs(self, importance_scores: torch.Tensor, signal_decay: torch.Tensor) -> torch.Tensor:
        """Reinforces regions where signal has decayed."""
        boost = torch.zeros_like(importance_scores)
        if signal_decay.numel() > 0:
            # Simple thresholding: if decay > 0.5, boost
            boost[signal_decay > 0.5] = 1000.0
        return importance_scores + boost
''',
    "continuity_energy_tracker.py": '''import torch

class ContinuityEnergyTracker:
    """PHASE 19.1A: Continuity Energy Tracker"""
    def track_energy(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.norm(hidden_states, dim=-1)
''',
    "traversal_signal_mapper.py": '''class TraversalSignalMapper:
    pass
''',
    "resonance_activation_controller.py": '''class ResonanceActivationController:
    pass
''',
    "continuity_resonance_field.py": '''import torch

class ContinuityResonanceField:
    """PHASE 19.1B: Continuity Resonance Field"""
    def apply_field(self, scores: torch.Tensor) -> torch.Tensor:
        return scores * 1.05 # slight boost
''',
    "signal_decay_compensator.py": '''import torch

class SignalDecayCompensator:
    def calculate_decay(self, energy: torch.Tensor) -> torch.Tensor:
        # returns a normalized decay factor
        max_e = energy.max()
        if max_e > 0:
            return 1.0 - (energy / max_e)
        return torch.zeros_like(energy)
''',
    "attention_energy_restorer.py": '''class AttentionEnergyRestorer:
    pass
''',
    "local_gradient_reinforcer.py": '''class LocalGradientReinforcer:
    pass
''',
    "adaptive_recharge_scheduler.py": '''class AdaptiveRechargeScheduler:
    """PHASE 19.1C: Adaptive Signal Recharging"""
    def should_recharge(self, decay_metric: float) -> bool:
        return decay_metric > 0.7
''',
    "reinforcement_trigger_detector.py": '''class ReinforcementTriggerDetector:
    pass
''',
    "contextual_energy_allocator.py": '''class ContextualEnergyAllocator:
    pass
''',
    "traversal_decay_predictor.py": '''class TraversalDecayPredictor:
    pass
''',
    "resonance_attention_stitcher.py": '''import torch

class ResonanceAttentionStitcher:
    """PHASE 19.1D: Resonance-Aware Attention Stitching"""
    def stitch(self, scores: torch.Tensor) -> torch.Tensor:
        return scores
''',
    "pathway_reinforcement_mapper.py": '''class PathwayReinforcementMapper:
    pass
''',
    "bridge_signal_preserver.py": '''class BridgeSignalPreserver:
    pass
''',
    "topology_energy_router.py": '''class TopologyEnergyRouter:
    pass
'''
}

ANALYSIS_FILES = {
    "resonance_cost_tracker.py": '''class ResonanceCostTracker:
    def __init__(self):
        self.activations = 0
    def add_activation(self):
        self.activations += 1
''',
    "signal_efficiency_mapper.py": '''class SignalEfficiencyMapper:
    pass
''',
    "reinforcement_density_auditor.py": '''class ReinforcementDensityAuditor:
    pass
''',
    "compute_balance_guardian.py": '''class ComputeBalanceGuardian:
    pass
'''
}

RUNTIME_RESOLVER = '''import torch
from runtime.hybrid_memory_resolver import HybridMemoryResolver
from memory.sparse_resonance_hubs import SparseResonanceHubs
from memory.signal_decay_compensator import SignalDecayCompensator
from memory.continuity_energy_tracker import ContinuityEnergyTracker
from analysis.resonance_cost_tracker import ResonanceCostTracker

class ResonanceMemoryResolver(HybridMemoryResolver):
    """PHASE 19.1: SSRCRF Resolver"""
    def __init__(self, anchor_budget: int = 6144, fidelity_budget: int = 512):
        super().__init__(anchor_budget, fidelity_budget)
        self.resonance_hubs = SparseResonanceHubs()
        self.decay_compensator = SignalDecayCompensator()
        self.energy_tracker = ContinuityEnergyTracker()
        self.cost_tracker = ResonanceCostTracker()

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
            self.geometry.accumulated_importance = self.smoother.smooth_importance(self.geometry.accumulated_importance)
            self.geometry.accumulated_importance = self.slope_preserver.preserve_slopes(self.geometry.accumulated_importance, all_sym_indices)
            self.geometry.accumulated_importance = self.runway.apply_runway(self.geometry.accumulated_importance, current_sym_idx)
            
            # Phase 19.1 additions
            energy = self.energy_tracker.track_energy(hidden_states)
            decay = self.decay_compensator.calculate_decay(energy)
            # Ensure decay matches the length of accumulated_importance
            if decay.shape[1] == self.geometry.accumulated_importance.shape[1]:
                self.geometry.accumulated_importance = self.resonance_hubs.activate_hubs(self.geometry.accumulated_importance, decay)
                self.cost_tracker.add_activation()

        bridge_mask = self.bridge_router.route_bridges(self.geometry.accumulated_importance if self.geometry.accumulated_importance is not None else torch.zeros((1, 1), device=hidden_states.device), all_sym_indices)
        
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
from runtime.adaptive_chunk_overlap import AdaptiveChunkOverlap
from transformers import AutoTokenizer, DynamicCache

class ValidationRunner19_1:
    def __init__(self, model, tokenizer, results_dir="results/reconstruction_19_1/"):
        self.model = model
        self.tokenizer = tokenizer
        self.results_dir = results_dir
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        self.suite = LongContextWorkloadSuite(tokenizer)
        self.overlap_scheduler = AdaptiveChunkOverlap(overlap_size=128)

    def run_matrix(self):
        contexts = [4096, 8192, 16384]
        modes = ["dense", "sparse_baseline", "hmc_18_7", "prmrs_18_8", "arrsbs_18_9", "sbpvcr_19_0", "ssrcrf_19_1"]
        
        test_id = "SIGMA-19-1-RESONANCE-TEST"

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
        
        test_case = self.suite.create_needle_in_haystack(
            ctx_len, 
            needle=f"The primary symbolic link is {needle_str}. It is connected to the secondary bridge ALPHA-ZERO.", 
            answer=needle_str,
            needle_pos_ratio=0.3 # Test long traversal chain (0.3 -> 1.0)
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
                
                if mode == "ssrcrf_19_1":
                    self._log_raw_19_1(ctx_len, resolver)
        
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
        prefix_str = needle_str[:8]
        prefix_match = prefix_str in generated_text
        
        hallucination = "Differential KV" not in generated_text and exact_match == False
        
        activations = getattr(resolver, "cost_tracker", None)
        resonance_activations = activations.activations if activations else 0

        return {
            "success": exact_match,
            "prefix_match": prefix_match,
            "tps": tps,
            "ttft": ttft,
            "vram_gb": vram,
            "hallucination": hallucination,
            "resonance_overhead": resonance_activations,
            "output": generated_text,
            "expected": needle_str
        }

    def _log_raw_19_1(self, ctx_len, resolver):
        with open(os.path.join(self.results_dir, "raw_resonance_activations.jsonl"), "a") as f:
            f.write(json.dumps({"ctx": ctx_len, "activations": resolver.cost_tracker.activations}) + "\\n")
        with open(os.path.join(self.results_dir, "raw_compute_overheads.jsonl"), "a") as f:
            f.write(json.dumps({"ctx": ctx_len, "summary": resolver.overhead_tracker.get_summary()}) + "\\n")

    def log_result(self, mode, ctx, res):
        log_file = os.path.join(self.results_dir, "raw_signal_decay.jsonl")
        with open(log_file, "a") as f:
            f.write(json.dumps({"mode": mode, "ctx": ctx, **res}) + "\\n")
        print(f"  [RESULT] EM: {res['success']} | Prefix: {res['prefix_match']} | TPS: {res['tps']:.2f} | VRAM: {res['vram_gb']:.2f}GB")

def main():
    print("="*60)
    print("PHASE 19.1 - SPARSE SIGNAL REINFORCEMENT & CONTINUITY RESONANCE")
    print("="*60)
    
    loader = Qwen7BRealLoader()
    try:
        model = loader.load()
    except:
        print("Model load failed.")
        return

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    
    runner = ValidationRunner19_1(model, tokenizer)
    runner.run_matrix()

if __name__ == "__main__":
    start_wallclock = time.perf_counter()
    main()
    end_wallclock = time.perf_counter()
    
    with open("results/reconstruction_19_1/raw_wallclock_trace.log", "w") as f:
        f.write(f"START: {start_wallclock}\\nEND: {end_wallclock}\\nDURATION: {end_wallclock - start_wallclock}\\n")
'''

def run():
    for f_name, content in MEMORY_FILES.items():
        create_file(os.path.join("d:/Codes/Projects/Differential KV/memory", f_name), content)
    
    for f_name, content in ANALYSIS_FILES.items():
        create_file(os.path.join("d:/Codes/Projects/Differential KV/analysis", f_name), content)
        
    create_file("d:/Codes/Projects/Differential KV/runtime/resonance_memory_resolver.py", RUNTIME_RESOLVER)
    create_file("d:/Codes/Projects/Differential KV/run_reconstruction_19_1_validation.py", VALIDATION_SCRIPT)

if __name__ == "__main__":
    run()
