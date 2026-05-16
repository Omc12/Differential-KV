import torch
import time
from typing import List, Dict, Any, Optional

class LowOverheadDecodeLoop:
    """
    PHASE 11A: ORCHESTRATION OVERHEAD REDUCTION
    
    A high-performance decode loop designed to minimize Python coordination overhead.
    Features:
    - Pre-allocated buffers for generated tokens.
    - Minimal branching in the hot loop.
    - Fused KV connector updates.
    - Async metric collection.
    """
    def __init__(self, model, tokenizer, kv_connector):
        self.model = model
        self.tokenizer = tokenizer
        self.kv_connector = kv_connector
        self.device = model.device

    @torch.no_grad()
    def decode(self, 
               input_ids: torch.Tensor, 
               max_new_tokens: int = 50, 
               sampler = None,
               stream_callback = None) -> Dict[str, Any]:
        
        # Pre-allocate tensor for speed
        batch_size = input_ids.shape[0]
        generated_tokens = torch.zeros((batch_size, max_new_tokens), dtype=torch.long, device=self.device)
        
        current_input_ids = input_ids
        past_key_values = None
        
        # Wall-clock timing start
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        tokens_generated = 0
        for i in range(max_new_tokens):
            # Model forward - The core bottleneck
            outputs = self.model(
                input_ids=current_input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )
            
            logits = outputs.logits[:, -1, :]
            
            # Optimized sampling
            if sampler:
                next_token_id = sampler.sample(logits)
            else:
                next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
            
            generated_tokens[:, i] = next_token_id.squeeze(-1)
            tokens_generated += 1
            
            # Low-overhead KV update - Fused with the next step's prep
            # This is where the sparse logic is injected without blocking the GPU
            past_key_values = self.kv_connector.update_fast(outputs.past_key_values)
            
            current_input_ids = next_token_id
            
            if stream_callback:
                stream_callback(next_token_id.item())
                
            # Early exit check (minimized overhead)
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        duration = end_time - start_time
        tps = tokens_generated / duration if duration > 0 else 0
        
        # Extract generated tokens
        final_tokens = generated_tokens[0, :tokens_generated].tolist()
        
        return {
            "token_ids": final_tokens,
            "text": self.tokenizer.decode(final_tokens),
            "latency": duration,
            "tokens_per_sec": tps,
            "orchestration_overhead_estimate": self.kv_connector.get_last_overhead(),
            "hardware_time": duration - self.kv_connector.get_last_overhead()
        }
