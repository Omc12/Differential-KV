
import torch
import time
from typing import Dict, List, Any
from transformers import AutoTokenizer, DynamicCache
from lcs.kv_pressure_profiler import KVPressureProfiler
from lcs.long_session_residency_tracker import LongSessionResidencyTracker
from lcs.sparse_scaling_curve_analyzer import SparseScalingCurveAnalyzer
from lcs.scaling_integrity_guard import ScalingIntegrityGuard
from rbe.gpu_telemetry_monitor import GPUTelemetryMonitor
from rbe.serving_latency_profiler import ServingLatencyProfiler

class LongContextBenchmarkOrchestrator:
    """
    PHASE 24.3: Long-Context Benchmark Orchestrator (LCS).
    Executes 4k–32k scaling benchmarks.
    """
    def __init__(self, model, tokenizer, config: Dict[str, Any]):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.pressure_profiler = KVPressureProfiler(config)
        self.residency_tracker = LongSessionResidencyTracker(config)
        self.curve_analyzer = SparseScalingCurveAnalyzer()
        self.integrity_guard = ScalingIntegrityGuard(config)
        
        # Reuse telemetry and profiler from RBE
        self.telemetry = GPUTelemetryMonitor()
        self.profiler = ServingLatencyProfiler()

    def run_scaling_benchmark(self, context_lengths: List[int], gen_tokens: int = 20):
        """
        Iterates through specified context lengths and collects scaling data.
        """
        print(f"Starting Long-Context Scaling Benchmark for {len(context_lengths)} points...")
        
        for clen in context_lengths:
            print(f"\n--- Context Length: {clen} ---")
            
            # 1. Run Dense Baseline (Simulated for long contexts to save time/VRAM if needed, 
            # but we'll try real if clen <= 16k)
            dense_metrics = self._run_pass(clen, gen_tokens, mode="dense")
            
            # 2. Run Sparse Pass
            sparse_metrics = self._run_pass(clen, gen_tokens, mode="sparse")
            
            # 3. Analyze and Record
            point = self.curve_analyzer.add_data_point(clen, dense_metrics, sparse_metrics)
            print(f"  TPS Ratio (Sparse/Dense): {point['tps_ratio']:.2f}x")
            print(f"  VRAM Savings: {point['vram_savings']*100:.1f}%")
            
        return self.curve_analyzer.analyze_scaling_trends()

    def _run_pass(self, context_len: int, gen_tokens: int, mode: str) -> Dict[str, float]:
        """
        Executes a single inference pass (prefill + generation).
        """
        request_id = f"{mode}_{context_len}"
        self.telemetry.start_session()
        self.profiler.start_request(request_id)
        
        # Load custom resolver if sparse
        resolver = None
        if mode == "sparse":
            from runtime.elf_resolver import ELFResolver
            resolver = ELFResolver(self.tokenizer)
            
        base_prompt = "Sparse cognition is the future of large-scale transformer serving. "
        # Generate a prompt of sufficient length
        prompt_len = 0
        p_parts = []
        while prompt_len < context_len - 100:
            p_parts.append(base_prompt)
            prompt_len += len(self.tokenizer.encode(base_prompt))
        full_prompt = "".join(p_parts)
        
        inputs = self.tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=context_len).to(self.model.device)
        input_ids = inputs.input_ids
        
        past_key_values = DynamicCache()
        
        # Prefill
        with torch.no_grad():
            outputs = self.model(input_ids, past_key_values=past_key_values, use_cache=True, output_hidden_states=True)
            if mode == "sparse" and resolver:
                resolver.resolve_and_prune(past_key_values, outputs.hidden_states[-1].detach(), input_ids)
            
        # Generation
        curr_input_ids = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
        
        for i in range(gen_tokens):
            with torch.no_grad():
                outputs = self.model(curr_input_ids, past_key_values=past_key_values, use_cache=True, output_hidden_states=True)
                if mode == "sparse" and resolver:
                    resolver.resolve_and_prune(past_key_values, outputs.hidden_states[-1].detach(), curr_input_ids)
                
            curr_input_ids = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
            self.profiler.record_token(request_id)
            
        metrics = self.profiler.get_metrics(request_id)
        gpu_metrics = self.telemetry.get_telemetry()
        
        return {
            "tps": metrics["tps"],
            "vram_gb": gpu_metrics.get("peak_vram_gb", 0.0),
            "latency_ms": metrics["avg_itl_ms"]
        }
