import torch
from transformers import AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import os
import time

class Qwen7BRealLoader:
    """
    MANDATORY PHASE 18.1A: Real-model loader for Qwen2.5-7B.
    FORBIDDEN: Proxy models or simulated weights.
    """
    def __init__(self, model_id: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.model_id = model_id

    def load(self, local_path: str = None, attn_implementation: str = "sdpa"):
        print(f"[PHASE 18.1A] Loading REAL Checkpoint: {self.model_id}")
        
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        try:
            # Note: local_files_only=True ensures we don't accidentally pull a proxy if offline
            model = AutoModelForCausalLM.from_pretrained(
                local_path if local_path else self.model_id,
                quantization_config=quant_config,
                device_map="cuda", # Force to GPU
                torch_dtype=torch.float16,
                trust_remote_code=True,
                local_files_only=True if not local_path else False
            )
            
            print(f"[SUCCESS] Real 7B Checkpoint loaded into VRAM.")
            return model
            
        except Exception as e:
            print(f"[CRITICAL FAILURE] Phase 18.1 load failed: {e}")
            print("Ensure that model weights are fully downloaded and not .incomplete.")
            raise e
