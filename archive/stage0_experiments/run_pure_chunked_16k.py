import torch
import gc
from models.qwen7b_real_loader import Qwen7BRealLoader
from transformers import AutoTokenizer, DynamicCache

def pure_chunked_test():
    print("[TEST] Loading model for pure chunked test...")
    loader = Qwen7BRealLoader()
    model = loader.load()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    
    ctx_len = 16384
    chunk_size = 512
    print(f"[RUN] 16k Context with {chunk_size} chunks (Native Attention)...")
    
    text = "Pure chunked prefill test." * (ctx_len // 5)
    prompt_ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=ctx_len).input_ids.to("cuda")
    
    past_key_values = DynamicCache()
    
    try:
        for i in range(0, ctx_len, chunk_size):
            chunk = prompt_ids[:, i:i+chunk_size]
            with torch.no_grad():
                # Process one chunk
                model(input_ids=chunk, past_key_values=past_key_values, use_cache=True)
            
            if i % 2048 == 0:
                print(f"  Processed {i}/{ctx_len}... VRAM: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
        
        print("[SUCCESS] 16k Context processed successfully!")
        
    except Exception as e:
        print(f"[FAILURE] 16k failed: {e}")

if __name__ == "__main__":
    pure_chunked_test()
