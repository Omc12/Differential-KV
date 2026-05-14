import torch
from models.qwen7b_real_loader import Qwen7BRealLoader
from transformers import AutoTokenizer

def isolation_test():
    print("[TEST] Loading model for isolation test...")
    loader = Qwen7BRealLoader()
    model = loader.load()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    
    print(f"[MEASURED] Base VRAM: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
    
    ctx_len = 4096
    text = "Isolation test for memory footprint." * (ctx_len // 10)
    prompt_ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=ctx_len).input_ids.to("cuda")
    
    with torch.no_grad():
        outputs = model(input_ids=prompt_ids, use_cache=False)
    
    print(f"[MEASURED] 4k Peak VRAM: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
    print(f"[MEASURED] CUDA Reserved: {torch.cuda.memory_reserved() / (1024**3):.2f} GB")

if __name__ == "__main__":
    isolation_test()
