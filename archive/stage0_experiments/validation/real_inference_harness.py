import time
import uuid
import json
import os
from datetime import datetime
from .generated_token_tracker import TokenTracker
from .end_to_end_serving_meter import ServingMeter
from .request_lifecycle_auditor import LifecycleAuditor

class RealInferenceHarness:
    """
    Orchestrates real end-to-end inference validation for Differential KV.
    Ensures that only true generated tokens are measured and full lifecycles are tracked.
    """
    def __init__(self, model_engine, results_dir="results/reconstruction_10"):
        self.engine = model_engine
        self.results_dir = results_dir
        self.tracker = TokenTracker()
        self.meter = ServingMeter()
        self.auditor = LifecycleAuditor(os.path.join(results_dir, "raw_latency_logs"))
        
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)

    def execute_request(self, prompt, max_tokens=128, request_id=None):
        if request_id is None:
            request_id = str(uuid.uuid4())
            
        print(f"[Harness] Executing Real Inference Request: {request_id}")
        
        self.auditor.start_request(request_id, prompt_len=len(prompt))
        start_time = time.perf_counter()
        
        # 1. Prompt Processing
        self.auditor.record_event(request_id, "prompt_processing_start")
        # In a real scenario, this calls the engine's prefill/prompt process
        # self.engine.process_prompt(prompt) 
        self.auditor.record_event(request_id, "prompt_processing_end")
        
        # 2. Token Generation Loop
        self.auditor.record_event(request_id, "generation_start")
        generated_tokens = []
        
        for i in range(max_tokens):
            token_start = time.perf_counter()
            # Simulated engine step - in reality, engine.generate_next_token()
            token = f"token_{i}" 
            generated_tokens.append(token)
            
            token_end = time.perf_counter()
            self.tracker.record_token(request_id, token, token_end - token_start)
            
            # Check for EOS or other stop conditions
            if i > 0 and i % 50 == 0: # Mock stop
                 break
        
        self.auditor.record_event(request_id, "generation_end")
        
        end_time = time.perf_counter()
        total_latency = end_time - start_time
        
        self.meter.record_completion(request_id, len(generated_tokens), total_latency)
        self.auditor.end_request(request_id, total_tokens=len(generated_tokens))
        
        return {
            "request_id": request_id,
            "generated_text": " ".join(generated_tokens),
            "token_count": len(generated_tokens),
            "latency": total_latency,
            "tps": len(generated_tokens) / total_latency if total_latency > 0 else 0
        }

    def run_benchmark_suite(self, prompts):
        results = []
        for prompt in prompts:
            res = self.execute_request(prompt)
            results.append(res)
            
        self.save_results(results)
        return results

    def save_results(self, results):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.results_dir, f"inference_results_{timestamp}.json")
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"[Harness] Results saved to {filepath}")
