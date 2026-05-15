import torch
import json
import time
import os
import random
from models.qwen7b_real_loader import Qwen7BRealLoader
from validation.multidomain_symbolic_suite import MultidomainSymbolicSuite
from validation.noisy_context_injector import NoisyContextInjector
from validation.realworld_context_generator import RealWorldContextGenerator
from validation.cross_run_consistency_checker import CrossRunConsistencyChecker
from validation.domain_generalization_tracker import DomainGeneralizationTracker
from validation.retrieval_consistency_monitor import RetrievalConsistencyMonitor
from analysis.failure_boundary_mapper import FailureBoundaryMapper
from analysis.context_stress_tester import ContextStressTester
from analysis.arbitration_instability_tracker import ArbitrationInstabilityTracker
from analysis.symbolic_degradation_profiler import SymbolicDegradationProfiler
from analysis.generalization_overhead_tracker import GeneralizationOverheadTracker
from runtime.calibrated_memory_resolver import CalibratedMemoryResolver
from runtime.hybrid_memory_resolver import HybridMemoryResolver
from transformers import AutoTokenizer, DynamicCache

# Correcting imports for internal modules
try:
    from validation.cross_run_consistency_checker import CrossRunConsistencyChecker
except ImportError:
    pass

class ValidationRunner20_0:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.symbolic_suite = MultidomainSymbolicSuite(tokenizer)
        self.noise_injector = NoisyContextInjector()
        self.rw_generator = RealWorldContextGenerator()
        self.results_dir = "results/reconstruction_20_0"
        os.makedirs(self.results_dir, exist_ok=True)
        self.log_file = os.path.join(self.results_dir, "raw_generalization_runs.jsonl")
        self.consistency_checker = CrossRunConsistencyChecker()
        self.failure_mapper = FailureBoundaryMapper(self.results_dir)
        self.gen_tracker = DomainGeneralizationTracker()
        self.retrieval_monitor = RetrievalConsistencyMonitor()
        self.stress_tester = ContextStressTester(self)
        self.instability_tracker = ArbitrationInstabilityTracker()
        self.degradation_profiler = SymbolicDegradationProfiler()
        self.overhead_tracker = GeneralizationOverheadTracker()

    def execute_single_run(self, mode, ctx_len, domain, use_noise=False, run_id=0):
        print(f"\n[RUNNING] Mode: {mode}, Context: {ctx_len}, Domain: {domain}, Noise: {use_noise}")
        
        # Setup Resolver
        if mode == "dense":
            resolver = None
        elif mode == "sparse_baseline":
            resolver = HybridMemoryResolver(anchor_budget=ctx_len//2, fidelity_budget=1024)
        elif mode in ["dtascc_19_7", "dtascc_20_0"]:
            resolver = CalibratedMemoryResolver(anchor_budget=ctx_len // 2, fidelity_budget=1024)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Create Test Case
        test_case = self.symbolic_suite.create_domain_test_case(domain, ctx_len)
        needle = test_case['needle']
        
        if use_noise:
            full_prompt = self.noise_injector.inject_noise(test_case['full_prompt'], intensity=0.1)
            input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt").to("cuda")
        else:
            input_ids = torch.tensor([test_case['tokens']]).to("cuda")

        past_key_values = DynamicCache()
        
        # Prefill
        start_prefill = time.perf_counter()
        chunk_size = 512
        for i in range(0, input_ids.shape[1], chunk_size):
            chunk = input_ids[:, i:i+chunk_size]
            with torch.no_grad():
                outputs = self.model(input_ids=chunk, past_key_values=past_key_values, use_cache=True, output_hidden_states=True)
                if resolver:
                    legacy = past_key_values.to_legacy_cache()
                    pruned, meta = resolver.resolve_and_prune(legacy, outputs.hidden_states[-1], chunk)
                    past_key_values = DynamicCache.from_legacy_cache(pruned)
        
        ttft = time.perf_counter() - start_prefill
        vram = torch.cuda.memory_allocated() / 1024**3

        # Generation
        print(f"Generating for {mode}...")
        curr_input = input_ids[:, -1:]
        generated_tokens = []
        start_gen = time.perf_counter()
        for _ in range(64):
            with torch.no_grad():
                outputs = self.model(input_ids=curr_input, past_key_values=past_key_values, use_cache=True)
                logits = outputs.logits[:, -1, :] / 0.7
                
                if mode in ["dtascc_19_7", "dtascc_20_0"]:
                    logits = resolver.guide_decoder(logits)
                
                token = torch.argmax(logits, dim=-1).unsqueeze(0)
                generated_tokens.append(token.item())
                curr_input = token
                
                if token.item() == self.tokenizer.eos_token_id:
                    break
        
        duration = time.perf_counter() - start_gen
        tps = len(generated_tokens) / duration if duration > 0 else 0
        output_text = self.tokenizer.decode(generated_tokens)
        
        # Validation
        def normalize(text):
            return "".join(text.lower().split())
            
        success = normalize(needle) in normalize(output_text)
        self.retrieval_monitor.log_retrieval(needle, output_text)
        self.gen_tracker.record_result(domain, success, tps)
        self.degradation_profiler.record_attempt(domain, ctx_len, success)
        
        trust_overhead = getattr(resolver, 'trust_overhead', None).total_alignment_time if hasattr(resolver, 'trust_overhead') and resolver.trust_overhead else 0.0
        self.overhead_tracker.record_metrics(mode, ctx_len, tps, vram, trust_overhead)
        
        result = {
            "mode": mode,
            "ctx": ctx_len,
            "domain": domain,
            "noise": use_noise,
            "run_id": run_id,
            "success": success,
            "tps": tps,
            "ttft": ttft,
            "vram_gb": vram,
            "output": output_text,
            "expected": needle,
            "trust_confidence": getattr(resolver, 'global_confidence', 0.0) if resolver else 1.0
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(result) + "\n")
            
        self.consistency_checker.add_run(run_id, result)
        print(f"[RESULT] Success: {success}, TPS: {tps:.2f}, VRAM: {vram:.2f}GB")
        return result

    def run_validation_suite(self):
        modes = ["dense", "sparse_baseline", "dtascc_19_7", "dtascc_20_0"]
        contexts = [4096, 8192, 16384]
        domains = ["activation_code", "api_key", "json_snippet", "code_fragment", "multilingual"]
        
        # Phase 20.0A & 20.0B
        for ctx in contexts:
            for domain in domains:
                for mode in modes:
                    # Run twice for reproducibility check (20.0D)
                    for r in range(2):
                        self.execute_single_run(mode, ctx, domain, use_noise=(r==1), run_id=r)

        # Final Analysis
        self.failure_mapper.analyze_results(self.log_file)
        self.failure_mapper.export_report()
        
        summary = self.gen_tracker.get_summary()
        with open(os.path.join(self.results_dir, "generalization_summary.json"), "w") as f:
            json.dump(summary, f, indent=4)
            
        consistency_report = self.consistency_checker.check_consistency()
        with open(os.path.join(self.results_dir, "reproducibility_report.json"), "w") as f:
            json.dump(consistency_report, f, indent=4)
        
        curves = self.degradation_profiler.get_degradation_curves()
        with open(os.path.join(self.results_dir, "degradation_curves.json"), "w") as f:
            json.dump(curves, f, indent=4)
            
        overhead = self.overhead_tracker.get_overhead_summary()
        with open(os.path.join(self.results_dir, "overhead_summary.json"), "w") as f:
            json.dump(overhead, f, indent=4)
            
        # Stress Testing (20.0C)
        self.stress_tester.run_stress_test("dtascc_20_0", base_ctx=16384)

if __name__ == "__main__":
    loader = Qwen7BRealLoader()
    model = loader.load()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    runner = ValidationRunner20_0(model, tokenizer)
    
    if os.path.exists(runner.log_file):
        os.remove(runner.log_file)
        
    runner.run_validation_suite()
