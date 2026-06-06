import os
import sys
import time
import gc
import torch
import psutil

# Ensure ACTIVE_RUNTIME is in path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["DIFFKV_USE_TORCH_COMPILE"] = "0"
os.environ["DIFFKV_TELEMETRY"] = "1"

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
    print("  MPS Memory Profiler (7700+ tokens)")
    print("=" * 60)
    
    # Disable capture_to_graph to prevent graph compilation leaks on dynamic shapes
    if hasattr(torch, "mps") and hasattr(torch.mps, "capture_to_graph"):
        print("[PROFILER] Deleting torch.mps.capture_to_graph to disable graph compilation")
        delattr(torch.mps, "capture_to_graph")
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load wrapper and model
    wrapper = DiffKVHFWrapper(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        config={"rank": 16, "micro_block_size": 16},
        device=device
    )
    
    # Replicate abstract to get ~7700 tokens
    abstract = (
        "While there is a growing effort towards AI for Sustainability (e.g. towards the sustainable development goals) "
        "it is time to move beyond that and to address the sustainability of developing and using AI systems. "
        "In this paper I propose a definition of Sustainable AI; Sustainable AI is a movement to foster change in "
        "the entire lifecycle of AI products (i.e. idea generation, training, re-tuning, implementation, governance, "
        "and post-use disposal) towards ecological and social sustainability. Sustainable AI is divided into two "
        "categories: AI for sustainability (using AI to support sustainability goals) and sustainability of AI "
        "(sustainable development, training, and use of AI). The focus of this paper is on the latter. "
        "In particular, I argue that the current trajectory of AI development and use (characterized by massive "
        "deep learning models requiring huge amounts of energy and resources to train and run) is unsustainable. "
        "I analyze the ecological and social impacts of the AI lifecycle, including resource extraction for hardware, "
        "greenhouse gas emissions from data centers during training and inference, and the social inequalities "
        "perpetuated by high compute costs. Finally, I propose a set of guiding principles and actionable "
        "recommendations for researchers, developers, and policymakers to transition towards a sustainable AI ecosystem. "
        "These include energy-efficient hardware, green software engineering, open data and models, and robust governance "
        "frameworks that incorporate environmental impact assessments. "
    )
    
    # ~250 tokens per abstract. Let's repeat 30 times to get ~7500 tokens.
    long_text = "\n".join([f"Paragraph {i+1}:\n{abstract}" for i in range(30)])
    prompt = f"<|im_start|>user\nHere is a long paper:\n{long_text}\n\nWhat is the main topic?<|im_end|>\n<|im_start|>assistant\n"
    
    encoded = wrapper.tokenizer(prompt, return_tensors="pt")
    num_tokens = encoded.input_ids.shape[1]
    print(f"Prompt tokens: {num_tokens}")
    
    engine = ContinuousBatchEngine(wrapper, max_batch_size=2)
    engine.start()
    
    print(f"Initial - MPS Mem: {get_mps_mem():.2f} MB | CPU Mem: {get_cpu_mem():.2f} MB")
    
    # Submit request
    session_id = "session_prof_7700"
    q = await engine.submit(session_id, {
        "prompt": prompt,
        "max_tokens": 30,  # generate 30 tokens
        "temperature": 0.0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
    })
    
    # We will manually drive the engine step-by-step or listen to its queue
    # Let's read from queue and log memory
    step = 0
    while True:
        chunk = await q.get()
        if "error" in chunk:
            print(f"\nEngine error: {chunk['error']}")
            break
        text = chunk.get("text", "")
        if text:
            # Perform GC and clear cache before measuring memory to see if memory is reclaimable
            gc.collect()
            if hasattr(torch, "mps") and torch.mps.is_available():
                torch.mps.empty_cache()
            
            print(f"Step {step:03d} | Token: {repr(text)} | MPS Mem: {get_mps_mem():.2f} MB | CPU Mem: {get_cpu_mem():.2f} MB")
            # Dump decode_workspace details
            ws = wrapper.manager.decode_workspace
            ws_size = 0.0
            ws_details = []
            for k, val in list(ws.items()):
                if isinstance(val, torch.Tensor):
                    sz = (val.numel() * val.element_size()) / (1024 ** 2)
                    ws_size += sz
                    ws_details.append(f"  Tensor Key: {k} | Shape: {list(val.shape)} | Size: {sz:.3f} MB")
                elif isinstance(val, tuple):
                    tuple_sz = 0.0
                    for i, t in enumerate(val):
                        if isinstance(t, torch.Tensor):
                            sz = (t.numel() * t.element_size()) / (1024 ** 2)
                            tuple_sz += sz
                            ws_details.append(f"  Tuple Key: {k}[{i}] | Shape: {list(t.shape)} | Size: {sz:.3f} MB")
                    ws_size += tuple_sz
                else:
                    ws_details.append(f"  Other Key: {k} | Type: {type(val)}")
            print(f"--> decode_workspace Total Size: {ws_size:.2f} MB (contains {len(ws)} keys)")
            # Print first few keys
            for detail in ws_details[:10]:
                print(detail)
            if len(ws_details) > 10:
                print(f"  ... and {len(ws_details) - 10} more tensors")
            
            step += 1
        if chunk.get("is_final"):
            break
            
    print(f"\nFinal - MPS Mem: {get_mps_mem():.2f} MB | CPU Mem: {get_cpu_mem():.2f} MB")
    await engine.stop()
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
