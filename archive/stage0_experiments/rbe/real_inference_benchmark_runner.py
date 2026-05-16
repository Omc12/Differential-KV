
import torch
import time
from typing import Dict, List, Any, Optional
from transformers import AutoTokenizer, DynamicCache
from rbe.gpu_telemetry_monitor import GPUTelemetryMonitor
from rbe.serving_latency_profiler import ServingLatencyProfiler

class RealInferenceBenchmarkRunner:
    """
    PHASE 24.2: Real Inference Benchmark Runner (RBE).
    Executes actual token generation and serving tests.
    """
    def __init__(self, model, tokenizer, config: Dict[str, Any]):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.telemetry = GPUTelemetryMonitor()
        self.profiler = ServingLatencyProfiler()
        self.device = next(model.parameters()).device
        
    def run_inference_test(self, 
                           prompt: str, 
                           max_tokens: int = 50, 
                           request_id: str = "req_0") -> Dict[str, Any]:
        """
        Runs a real single-stream inference pass and records telemetry.
        """
        self.telemetry.start_session()
        self.profiler.start_request(request_id)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        
        # We use standard DynamicCache for dense baseline comparison if needed,
        # or custom resolver logic for sparse.
        past_key_values = DynamicCache()
        
        generated_tokens = []
        
        for i in range(max_tokens):
            k_event = self.telemetry.record_kernel_start()
            
            with torch.no_grad():
                outputs = self.model(input_ids, past_key_values=past_key_values, use_cache=True)
                
            self.telemetry.record_kernel_end(k_event)
            
            logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(logits, dim=-1).unsqueeze(0)
            
            input_ids = next_token_id
            generated_tokens.append(next_token_id.item())
            self.profiler.record_token(request_id)
            
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break
                
        metrics = self.profiler.get_metrics(request_id)
        gpu_metrics = self.telemetry.get_telemetry()
        
        return {
            "tokens": generated_tokens,
            "text": self.tokenizer.decode(generated_tokens),
            "metrics": metrics,
            "gpu_metrics": gpu_metrics
        }

    def run_concurrent_serving(self, 
                               prompts: List[str], 
                               max_tokens: int = 50) -> List[Dict[str, Any]]:
        """
        Simulates concurrent serving requests.
        """
        # In a real environment, this might use threading or async.
        # For the benchmark, we'll run them sequentially but track metrics.
        results = []
        for i, prompt in enumerate(prompts):
            res = self.run_inference_test(prompt, max_tokens, f"concurrent_{i}")
            results.append(res)
        return results
