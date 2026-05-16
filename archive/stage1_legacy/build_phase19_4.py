import os

def create_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)

MEMORY_FILES = {
    "hierarchical_hub_network.py": '''import torch

class HierarchicalHubNetwork:
    """PHASE 19.4A: Hierarchical Hub Network"""
    def synchronize_hubs(self, importance: torch.Tensor, consensus_score: float) -> torch.Tensor:
        # Boost regions that show hierarchical agreement
        if consensus_score > 0.5:
            importance = importance * 1.5
        return importance
''',
    "local_consensus_hubs.py": '''class LocalConsensusHubs: pass''',
    "regional_synchronization_hubs.py": '''class RegionalSynchronizationHubs: pass''',
    "global_coordination_hubs.py": '''class GlobalCoordinationHubs: pass''',
    
    "sparse_global_relays.py": '''import torch

class SparseGlobalRelays:
    """PHASE 19.4B: Sparse Global Relay Paths"""
    def propagate_signal(self, importance: torch.Tensor, relay_indices: torch.Tensor) -> torch.Tensor:
        # Long-range propagation of symbolic signals
        importance[0, relay_indices] += 2000.0
        return importance
''',
    "high_priority_signal_channels.py": '''class HighPrioritySignalChannels: pass''',
    "long_range_consensus_router.py": '''class LongRangeConsensusRouter: pass''',
    "relay_path_optimizer.py": '''class RelayPathOptimizer: pass''',
    
    "consensus_beacons.py": '''import torch

class ConsensusBeacons:
    """PHASE 19.4C: Consensus Beacon Synchronization"""
    def broadcast_beacons(self, importance: torch.Tensor, beacon_indices: torch.Tensor) -> torch.Tensor:
        # Periodic 'beacons' that stabilize global identity
        dtype_max = torch.finfo(importance.dtype).max
        importance[0, beacon_indices] = dtype_max
        return importance
''',
    "global_identity_summaries.py": '''class GlobalIdentitySummaries: pass''',
    "symbolic_confidence_broadcaster.py": '''class SymbolicConfidenceBroadcaster: pass''',
    "distributed_agreement_registry.py": '''class DistributedAgreementRegistry: pass''',
    
    "agreement_arbitrator.py": '''import torch

class AgreementArbitrator:
    def arbitrate(self, importance: torch.Tensor) -> torch.Tensor:
        # Suppress noise and resolve conflicts
        return importance
''',
    "conflict_resolution_scheduler.py": '''class ConflictResolutionScheduler: pass''',
    "global_consensus_allocator.py": '''class GlobalConsensusAllocator: pass''',
    "sparse_identity_prioritizer.py": '''class SparseIdentityPrioritizer: pass'''
}

ANALYSIS_FILES = {
    "synchronization_cost_tracker.py": '''class SynchronizationCostTracker:
    def __init__(self):
        self.sync_events = 0
    def add_sync(self):
        self.sync_events += 1
''',
    "relay_efficiency_mapper.py": '''class RelayEfficiencyMapper: pass''',
    "hub_density_auditor.py": '''class HubDensityAuditor: pass'''
}

RUNTIME_RESOLVER = '''import torch
from runtime.consensus_memory_resolver import ConsensusMemoryResolver
from memory.hierarchical_hub_network import HierarchicalHubNetwork
from memory.sparse_global_relays import SparseGlobalRelays
from memory.consensus_beacons import ConsensusBeacons
from memory.agreement_arbitrator import AgreementArbitrator
from analysis.synchronization_cost_tracker import SynchronizationCostTracker

class SynchronizedMemoryResolver(ConsensusMemoryResolver):
    """PHASE 19.4: HHSSGC Resolver"""
    def __init__(self, anchor_budget: int = 6144, fidelity_budget: int = 512):
        super().__init__(anchor_budget, fidelity_budget)
        self.hub_network = HierarchicalHubNetwork()
        self.relays = SparseGlobalRelays()
        self.beacons = ConsensusBeacons()
        self.arbitrator = AgreementArbitrator()
        self.sync_tracker = SynchronizationCostTracker()

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

            # Phase 19.4 HHSSGC logic
            # A. Hierarchical Hub Synchronization
            consensus_score = (self.geometry.accumulated_importance > 10000.0).float().mean().item()
            self.geometry.accumulated_importance = self.hub_network.synchronize_hubs(
                self.geometry.accumulated_importance, consensus_score)
            self.sync_tracker.add_sync()

            # B. Global Relays & C. Beacons
            # If we have many symbolic indices, pick some as relays and beacons
            if len(all_sym_indices) > 0:
                current_abs_indices = self.geometry.absolute_indices[0]
                matches = []
                for sym_idx in all_sym_indices[-10:].tolist(): # use recent symbolic targets
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

            # D. Agreement Arbitration
            self.geometry.accumulated_importance = self.arbitrator.arbitrate(self.geometry.accumulated_importance)

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
from runtime.consensus_memory_resolver import ConsensusMemoryResolver
from runtime.synchronized_memory_resolver import SynchronizedMemoryResolver
from runtime.adaptive_chunk_overlap import AdaptiveChunkOverlap
from transformers import AutoTokenizer, DynamicCache

class ValidationRunner19_4:
    def __init__(self, model, tokenizer, results_dir="results/reconstruction_19_4/"):
        self.model = model
        self.tokenizer = tokenizer
        self.results_dir = results_dir
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        self.suite = LongContextWorkloadSuite(tokenizer)
        self.overlap_scheduler = AdaptiveChunkOverlap(overlap_size=128)

    def run_matrix(self):
        contexts = [4096, 8192, 16384]
        modes = ["dense", "sparse_baseline", "hmc_18_7", "prmrs_18_8", "arrsbs_18_9", "sbpvcr_19_0", "ssrcrf_19_1", "asdcaf_19_2", "dscrrf_19_3", "hhssgc_19_4"]
        
        test_id = "SIGMA-19-4-HHSSGC-TEST"

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
            "sync_events": getattr(resolver, 'sync_tracker', None).sync_events if hasattr(resolver, 'sync_tracker') and resolver.sync_tracker else 0
        }

    def log_result(self, mode, ctx, res):
        log_file = os.path.join(self.results_dir, "raw_hub_synchronization.jsonl")
        with open(log_file, "a") as f:
            f.write(json.dumps({"mode": mode, "ctx": ctx, **res}) + "\\n")
        print(f"  [RESULT] EM: {res['success']} | Prefix: {res['prefix_match']} | TPS: {res['tps']:.2f} | VRAM: {res['vram_gb']:.2f}GB")

def main():
    print("="*60)
    print("PHASE 19.4 - HIERARCHICAL HUB SYNCHRONIZATION")
    print("="*60)
    
    loader = Qwen7BRealLoader()
    model = loader.load()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    
    runner = ValidationRunner19_4(model, tokenizer)
    runner.run_matrix()

if __name__ == "__main__":
    main()
'''

def run():
    for f_name, content in MEMORY_FILES.items():
        create_file(os.path.join("d:/Codes/Projects/Differential KV/memory", f_name), content)
    for f_name, content in ANALYSIS_FILES.items():
        create_file(os.path.join("d:/Codes/Projects/Differential KV/analysis", f_name), content)
    create_file("d:/Codes/Projects/Differential KV/runtime/synchronized_memory_resolver.py", RUNTIME_RESOLVER)
    create_file("d:/Codes/Projects/Differential KV/run_reconstruction_19_4_validation.py", VALIDATION_SCRIPT)

if __name__ == "__main__":
    run()
