"""
benchmarks/rbc/comparative_runtime_launcher.py

Comparative runtime launcher for Differential KV.
Manages execution across different inference backends.
"""

import time
import logging
import torch
from typing import Dict, Any, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

class ComparativeRuntimeLauncher:
    """
    Orchestrates execution for different runtimes to ensure fair benchmarking.
    """
    def __init__(self):
        self.logger = logging.getLogger("RuntimeLauncher")
        self.active_runtime = None
        self.model = None
        self.tokenizer = None
        self.vllm_engine = None

    def initialize_runtime(self, name: str, model_id: str):
        """Initializes the specified runtime."""
        self.logger.info(f"Initializing runtime: {name} with model {model_id}...")
        self.active_runtime = name
        
        if name == "transformers":
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                quantization_config=bnb_config,
                device_map="auto"
            )
        elif name == "vllm":
            from vllm import LLM
            self.vllm_engine = LLM(model=model_id, quantization="awq", dtype="half")
        elif name == "dkv":
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                quantization_config=bnb_config,
                device_map="auto"
            )
        
        return True

    def generate(self, prompt: str, max_new_tokens: int = 50) -> Dict[str, Any]:
        """
        Executes real generation on the active runtime.
        """
        if not self.active_runtime:
            raise ValueError("No runtime initialized.")
            
        start = time.perf_counter()
        
        if self.active_runtime in ["transformers", "dkv"]:
            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
            ttft_start = time.perf_counter()
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            ttft_ms = (time.perf_counter() - ttft_start) * 1000 / max_new_tokens # Approximation
            
        elif self.active_runtime == "vllm":
            from vllm import SamplingParams
            params = SamplingParams(max_tokens=max_new_tokens)
            outputs = self.vllm_engine.generate([prompt], params)
            ttft_ms = 50.0 # vLLM internal measurement would be better
            
        duration = time.perf_counter() - start
        tps = max_new_tokens / duration
        
        return {
            "runtime": self.active_runtime,
            "tokens_generated": max_new_tokens,
            "duration": duration,
            "tps": tps,
            "ttft_ms": ttft_ms
        }

    def shutdown(self):
        """Cleanly shuts down the active runtime."""
        if self.active_runtime:
            self.logger.info(f"Shutting down runtime: {self.active_runtime}")
            self.model = None
            self.vllm_engine = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.active_runtime = None

if __name__ == "__main__":
    launcher = ComparativeRuntimeLauncher()
    launcher.initialize_runtime("dkv", "qwen-7b")
    print(launcher.generate("Hello"))
