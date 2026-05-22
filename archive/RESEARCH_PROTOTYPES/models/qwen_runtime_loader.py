import torch
from transformers import AutoModelForCausalLM, AutoConfig
from .quantized_checkpoint_manager import QuantizedCheckpointManager
import time
import os

class QwenRuntimeLoader:
    """
    MANDATORY PHASE 18A: Real-model loader for Qwen2.5 family.
    Ensures all execution happens on physical pretrained weights.
    """
    def __init__(self, model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.model_id = model_id
        self.quant_manager = QuantizedCheckpointManager()

    def load(self, device="cuda", use_flash_attn=True):
        print(f"[PHASE 18A] Loading REAL pretrained model: {self.model_id}")
        start_time = time.time()
        
        quantization_config = self.quant_manager.get_config()
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=quantization_config,
                device_map=device,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation="flash_attention_2" if use_flash_attn else "sdpa",
                local_files_only=True
            )
            
            load_time = time.time() - start_time
            print(f"[SUCCESS] {self.model_id} loaded in {load_time:.2f}s")
            
            # Record manifest for reproducibility
            from .runtime_manifest_exporter import RuntimeManifestExporter
            exporter = RuntimeManifestExporter()
            exporter.record_load(self.model_id, quantization_config, load_time)
            
            return model
            
        except Exception as e:
            print(f"[ERROR] Failed to load real model {self.model_id}: {e}")
            print("[FALLBACK] Check if model is downloaded or if CUDA OOM occurred.")
            raise e

if __name__ == "__main__":
    # Test loader (dry run or verification)
    loader = QwenRuntimeLoader()
    print("Loader initialized for Phase 18.")
