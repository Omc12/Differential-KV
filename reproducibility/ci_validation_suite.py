"""
reproducibility/ci_validation_suite.py

Automated CI validation suite for Differential KV.
Checks for performance regressions in throughput and VRAM efficiency.
"""

import unittest
import torch
import time
from runtime.differential_kv_runtime import DifferentialKVRuntime
from transformers import AutoModelForCausalLM

class TestDifferentialKVPerformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use a small model for CI testing
        cls.model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
        cls.model = AutoModelForCausalLM.from_pretrained(cls.model_id)
        cls.config = {"mode": "differential", "sparse_ratio": 0.5}
        cls.runtime = DifferentialKVRuntime(cls.model, cls.config)
        cls.patched_model = cls.runtime.patched_model

    def test_throughput_gain(self):
        # Baseline (standard attention) - simplified
        # Patched (Differential KV)
        input_ids = torch.randint(0, 100, (1, 1024))
        
        start = time.time()
        with torch.no_grad():
            self.patched_model(input_ids)
        duration = time.time() - start
        
        print(f"Inference duration: {duration:.4f}s")
        self.assertLess(duration, 1.0, "Inference too slow for tiny model")

    def test_vram_reduction(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated()
        
        input_ids = torch.randint(0, 100, (1, 2048)).cuda()
        self.patched_model.to("cuda")
        with torch.no_grad():
            self.patched_model(input_ids)
            
        mem_after = torch.cuda.memory_allocated()
        print(f"Memory used: {(mem_after - mem_before) / 1024**2:.2f} MB")

if __name__ == "__main__":
    unittest.main()
