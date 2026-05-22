import torch
from models.qwen7b_real_loader import Qwen7BRealLoader
from runtime.guided_memory_resolver import GuidedMemoryResolver
from transformers import AutoTokenizer, DynamicCache
import time

def test_single_mode():
    loader = Qwen7BRealLoader()
    model = loader.load()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    
    ctx_len = 4096
    resolver = GuidedMemoryResolver(anchor_budget=ctx_len // 2, fidelity_budget=1024)
    
    input_ids = torch.randint(0, 32000, (1, 512)).to("cuda")
    past_key_values = DynamicCache()
    
    print("Prefilling...")
    with torch.no_grad():
        outputs = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True, output_hidden_states=True)
        resolver.resolve_and_prune(past_key_values.to_legacy_cache(), outputs.hidden_states[-1], input_ids)
    
    print("Generating...")
    curr_input = input_ids[:, -1:]
    logits = model(input_ids=curr_input, past_key_values=past_key_values, use_cache=True).logits[:, -1, :]
    
    try:
        print("Guiding decoder...")
        guided_logits = resolver.guide_decoder(logits)
        print("Success!")
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    test_single_mode()
