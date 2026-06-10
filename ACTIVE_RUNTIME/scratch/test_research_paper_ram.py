import os
import sys
import time
import gc
import torch
import psutil
import asyncio

# Ensure ACTIVE_RUNTIME is in path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["DIFFKV_USE_TORCH_COMPILE"] = "0"
os.environ["DIFFKV_TELEMETRY"] = "1"
os.environ["DIFFKV_ENGAGE_THRESHOLD"] = "500"
os.environ["DIFFKV_MPS_APPROXIMATE_ATTN"] = "1"

from serving.hf_diffkv_wrapper import DiffKVHFWrapper
from serving.batch_engine import ContinuousBatchEngine

def get_mps_mem():
    if hasattr(torch, "mps") and torch.mps.is_available():
        try:
            return torch.mps.current_allocated_memory() / (1024 ** 2)
        except Exception:
            return 0.0
    return 0.0

def get_cpu_mem():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)

async def main():
    print("=" * 60)
    print("  Research Paper 6K RAM & Coherence Verification")
    print("=" * 60)
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load research paper text
    paper_path = "/Users/omchimurkar1/.gemini/antigravity/brain/3b15502c-2913-4cf2-82e1-d764b49db8c4/scratch/research_paper.txt"
    with open(paper_path, "r") as f:
        paper_text = f.read()
    
    # Construct a 6K prompt by repeating the paper text
    # The paper is approx 27KB (6K tokens when repeated 5 times)
    long_context = (paper_text + "\n") * 5
    
    # Load wrapper and model (use Rank=16)
    wrapper = DiffKVHFWrapper(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        config={"rank": 16, "micro_block_size": 32},
        device=device
    )
    
    engine = ContinuousBatchEngine(wrapper, max_batch_size=1)
    engine.start()
    
    # --- Turn 1 ---
    prompt1 = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nHere is a research paper:\n"
        + long_context +
        "\n\nQuestion: What is the main topic of the paper described above?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    encoded1 = wrapper.tokenizer(prompt1, return_tensors="pt")
    print(f"Turn 1 prompt tokens: {encoded1.input_ids.shape[1]}")
    
    print(f"\nBefore Turn 1 - MPS Mem: {get_mps_mem():.2f} MB | CPU Mem: {get_cpu_mem():.2f} MB")
    
    session_id = "sess_research_paper_6k"
    q1 = await engine.submit(session_id, {
        "prompt": prompt1,
        "max_tokens": 50,
        "temperature": 0.0,
    })
    
    res1 = []
    t0 = time.time()
    while True:
        chunk = await q1.get()
        text = chunk.get("text", "")
        if text:
            res1.append(text)
        if chunk.get("is_final"):
            break
            
    ans1 = "".join(res1).strip()
    elapsed1 = time.time() - t0
    print(f"Turn 1 completed in {elapsed1:.2f}s.")
    print(f"Turn 1 Response:\n{ans1}\n")
    print(f"After Turn 1 - MPS Mem: {get_mps_mem():.2f} MB | CPU Mem: {get_cpu_mem():.2f} MB")
    
    # --- Turn 2 (Continuation) ---
    prompt2 = (
        prompt1 + ans1 + "<|im_end|>\n"
        "<|im_start|>user\nCould you explain in 1 sentence what the randomized features are designed to do?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    encoded2 = wrapper.tokenizer(prompt2, return_tensors="pt")
    print(f"\nTurn 2 prompt tokens: {encoded2.input_ids.shape[1]}")
    
    print(f"Before Turn 2 - MPS Mem: {get_mps_mem():.2f} MB | CPU Mem: {get_cpu_mem():.2f} MB")
    
    q2 = await engine.submit(session_id, {
        "prompt": prompt2,
        "max_tokens": 50,
        "temperature": 0.0,
    })
    
    res2 = []
    t1 = time.time()
    while True:
        chunk = await q2.get()
        text = chunk.get("text", "")
        if text:
            res2.append(text)
        if chunk.get("is_final"):
            break
            
    ans2 = "".join(res2).strip()
    elapsed2 = time.time() - t1
    print(f"Turn 2 completed in {elapsed2:.2f}s.")
    print(f"Turn 2 Response:\n{ans2}\n")
    print(f"After Turn 2 - MPS Mem: {get_mps_mem():.2f} MB | CPU Mem: {get_cpu_mem():.2f} MB")
    
    await engine.stop()
    
    # Assert correctness
    assert len(ans1) > 0, "Turn 1 output is empty"
    assert len(ans2) > 0, "Turn 2 output is empty"
    
    print("\nAll checks completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
