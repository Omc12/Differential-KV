import asyncio
import uvicorn
import torch
from runtime.lgs_resolver import LGSResolver
from serving.openai_compatible_api_gateway import OpenAICompatibleAPIGateway

config = {
    "lgs": {
        "max_ttft_ms": 15000,
        "max_itl_ms": 1000,
        "min_fairness_index": 0.8,
        "min_sparse_ratio": 0.9,
        "max_queue_wait_ms": 20000
    }
}

class SKO7BResolver(LGSResolver):
    def setup_runtime(self):
        from runtime.hf_diffkv_wrapper import DiffKVHFWrapper
        from decode_pipeline_fusion_engine import DecodePipelineFusionEngine
        
        if self.wrapper is None:
            model_id = "Qwen/Qwen2.5-7B-Instruct"
            print(f"[*] SKO PHASE: Loading REAL 7B Model -> {model_id}")
            # Use 4-bit if possible, otherwise FP16 with offload
            # Since bitsandbytes isn't in requirements, we'll try FP16
            self.wrapper = DiffKVHFWrapper(model_id, {
                "mode": "lowrank_sparse", 
                "block_size": 64, 
                "rank": 16
            })
            self.fusion_engine = DecodePipelineFusionEngine(self.wrapper)

import os
import json

startup_vram = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0

print("Initializing SKO 7B Resolver...")
resolver = SKO7BResolver(config)
resolver.setup_runtime()

post_load_vram = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0

# Persist server info as requested
server_info = {
    "model_name": resolver.wrapper.model_id,
    "dtype": str(resolver.wrapper.model.dtype),
    "device": str(resolver.wrapper.device),
    "startup_vram_gb": startup_vram,
    "post_load_vram_gb": post_load_vram,
    "quantization": "None (FP16)" # As observed in previous run
}

os.makedirs("telemetry/stage2/phase_38_6_sko", exist_ok=True)
with open("telemetry/stage2/phase_38_6_sko/server_info.json", "w") as f:
    json.dump(server_info, f, indent=4)

# Print live model info as requested
print(f"\n[SERVER-INFO] MODEL: {server_info['model_name']}")
print(f"[SERVER-INFO] DTYPE: {server_info['dtype']}")
print(f"[SERVER-INFO] DEVICE: {server_info['device']}")
print(f"[SERVER-INFO] STARTUP VRAM: {server_info['startup_vram_gb']:.2f} GB")
print(f"[SERVER-INFO] POST-LOAD VRAM: {server_info['post_load_vram_gb']:.2f} GB")

gateway = OpenAICompatibleAPIGateway(resolver)

if __name__ == "__main__":
    print("Starting SKO 7B Real Serving Stack (LIVE STREAMING MODE)...")
    uvicorn.run(gateway.app, host="0.0.0.0", port=8000)
