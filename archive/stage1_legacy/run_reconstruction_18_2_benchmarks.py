import torch
import time
import json
import os
from models.qwen7b_real_loader import Qwen7BRealLoader
from memory.real_sparse_kv_manager import RealSparseKVManager
from telemetry.strict_metric_taxonomy import StrictMetricTaxonomy
from telemetry.wallclock_enforcer import WallclockEnforcer
from analysis.sparse_failure_mapper import SparseFailureMapper

class SparseBenchmarkRunner:
    def __init__(self, model_id="Qwen/Qwen2.5-7B-Instruct"):
        self.loader = Qwen7BRealLoader(model_id)
        self.model = self.loader.load()
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.taxonomy = StrictMetricTaxonomy()
        self.enforcer = WallclockEnforcer()
        self.kv_manager = RealSparseKVManager(anchor_budget=2048, recent_window=1024)
        self.failure_mapper = SparseFailureMapper()

    def chunked_generate(self, prompt_ids, max_new_tokens=32, chunk_size=2048):
        """
        Executes generation with chunked prefill and sparse KV pruning.
        This is the core 'recovery' logic.
        """
        batch_size, seq_len = prompt_ids.shape
        past_key_values = None
        
        self.enforcer.start()
        
        # 1. Chunked Prefill (to avoid OOM at 16k)
        for i in range(0, seq_len, chunk_size):
            chunk = prompt_ids[:, i:i+chunk_size]
            outputs = self.model(
                input_ids=chunk,
                past_key_values=past_key_values,
                use_cache=True
            )
            past_key_values = outputs.past_key_values
            
            # Prune KV cache after each chunk to maintain budget
            past_key_values, _ = self.kv_manager.prune_kv(past_key_values)
            
        # 2. Autoregressive Decode
        input_ids = prompt_ids
        generated_tokens = 0
        
        for _ in range(max_new_tokens):
            outputs = self.model(
                input_ids=input_ids[:, -1:],
                past_key_values=past_key_values,
                use_cache=True
            )
            past_key_values = outputs.past_key_values
            
            # Regular Sparse Pruning
            past_key_values, _ = self.kv_manager.prune_kv(past_key_values)
            
            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            generated_tokens += 1
            
        duration = self.enforcer.stop()
        
        return {
            "tps": generated_tokens / duration,
            "vram_gb": torch.cuda.memory_allocated() / (1024**3),
            "tokens": generated_tokens,
            "duration": duration
        }

    def run_matrix(self):
        contexts = [4096, 8192, 16384]
        results = []
        
        print("="*60)
        print("PHASE 18.2 — REAL SPARSE KV RECOVERY")
        print("="*60)

        for ctx_len in contexts:
            print(f"\n[RUN] Context: {ctx_len} (Sparse)")
            
            # Prepare Input
            text = "Differential KV uses sparse anchors to maintain long-context retrieval integrity." * (ctx_len // 10)
            prompt_ids = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=ctx_len).input_ids.to("cuda")

            try:
                res = self.chunked_generate(prompt_ids)
                print(f"[RESULT] {self.taxonomy.log_measured('TPS', res['tps'])}")
                print(f"[RESULT] {self.taxonomy.log_measured('VRAM', res['vram_gb'], 'GB')}")
                
                results.append({
                    "context": ctx_len,
                    "tps": res["tps"],
                    "vram_gb": res["vram_gb"],
                    "status": "SUCCESS"
                })
            except Exception as e:
                print(f"[FAILURE] Sparse Context {ctx_len} failed: {e}")
                self.failure_mapper.record_failure(ctx_len, str(e))
                results.append({
                    "context": ctx_len,
                    "status": "FAILED",
                    "error": str(e)
                })
                torch.cuda.empty_cache()

        self.export_results(results)

    def export_results(self, results):
        path = "results/reconstruction_18_2/bench_results.json"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(results, f, indent=4)
        
        # Generate Markdown Report
        report_path = "results/reconstruction_18_2/reconstruction_18_2_sparse_recovery.md"
        with open(report_path, 'w') as f:
            f.write("# PHASE 18.2 — REAL SPARSE KV RECOVERY REPORT\n\n")
            f.write("| Context | Dense 18.1 | Sparse 18.2 [MEASURED] | Recovery Status |\n")
            f.write("|---|---|---|---|\n")
            
            # 18.1 Data for comparison
            dense_data = {4096: "7.04", 8192: "0.09", 16384: "OOM"}
            
            for r in results:
                d_tps = dense_data.get(r['context'], "N/A")
                s_tps = f"{r['tps']:.2f}" if r['status'] == "SUCCESS" else "FAILED"
                status = "RECOVERED" if r['status'] == "SUCCESS" and d_tps in ["0.09", "OOM"] else "STABLE"
                f.write(f"| {r['context']} | {d_tps} | {s_tps} | {status} |\n")

if __name__ == "__main__":
    # Note: SparseFailureMapper needs to be created
    if not os.path.exists("analysis/sparse_failure_mapper.py"):
        with open("analysis/sparse_failure_mapper.py", "w") as f:
            f.write("class SparseFailureMapper:\n    def record_failure(self, ctx, err): pass\n")
            
    runner = SparseBenchmarkRunner()
    runner.run_matrix()
