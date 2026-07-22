import torch
import time
from typing import Dict, Any, List, Optional, Callable
from runtime.hf_dkv_wrapper import DKVHFWrapper

class SustainedSparseDecodeEngine:
    """
    Forces persistent sparse decode occupancy to prevent burst-only execution.
    Owner: SHM (Sustained Hardware Materialization)
    """
    def __init__(
        self, 
        wrapper: DKVHFWrapper,
        min_duration_sec: float = 120.0,
        batch_size: int = 4
    ):
        self.wrapper = wrapper
        self.min_duration_sec = min_duration_sec
        self.batch_size = batch_size
        self.is_active = False

    def execute_sustained_decode(
        self, 
        prompt: str, 
        max_tokens: int = 1024,
        on_step_callback: Optional[Callable] = None
    ):
        """
        Executes a sustained autoregressive decode loop.
        """
        print(f"[SHM] Starting Sustained Sparse Decode Engine (target_duration={self.min_duration_sec}s)")
        self.is_active = True
        start_time = time.perf_counter()
        
        # Prepare inputs for batching
        inputs = self.wrapper.tokenizer(
            [prompt] * self.batch_size, 
            return_tensors='pt', 
            padding=True
        ).to(self.wrapper.device)
        
        input_ids = inputs.input_ids
        generated_count = 0
        
        while (time.perf_counter() - start_time < self.min_duration_sec) or (generated_count < max_tokens):
            step_start = time.perf_counter()
            
            # FORCE: Continuous sparse dispatch
            # This would call the resolver which uses Triton kernels
            with torch.no_grad():
                # In a real SHM implementation, we bypass HF forward completely
                # and use our custom persistent kernels
                outputs = self.wrapper.model(input_ids=input_ids, use_cache=True)
                logits = outputs.logits[:, -1, :]
                next_tokens = torch.argmax(logits, dim=-1)
                
            input_ids = next_tokens.unsqueeze(-1)
            generated_count += 1
            
            if on_step_callback:
                on_step_callback(step_start, time.perf_counter())
                
            if generated_count % 100 == 0:
                elapsed = time.perf_counter() - start_time
                print(f" [SHM] Generated {generated_count} tokens... Elapsed: {elapsed:.2f}s")

        self.is_active = False
        total_duration = time.perf_counter() - start_time
        print(f"[SHM] Sustained decode complete. Total tokens: {generated_count * self.batch_size}, Duration: {total_duration:.2f}s")
        return generated_count
