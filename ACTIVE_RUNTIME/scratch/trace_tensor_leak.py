import os
import sys
import gc
import torch

# Ensure ACTIVE_RUNTIME is in path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["DIFFKV_USE_TORCH_COMPILE"] = "0"
os.environ["DIFFKV_TELEMETRY"] = "0"

from serving.hf_diffkv_wrapper import DiffKVHFWrapper
from serving.batch_engine import ContinuousBatchEngine

def get_live_tensors():
    tensors = []
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor):
                tensors.append(obj)
        except Exception:
            pass
    return tensors

def print_tensor_diff(before, after, label):
    # Group by shape and dtype
    def get_stats(tensor_list):
        stats = {}
        for t in tensor_list:
            try:
                key = (tuple(t.shape), str(t.dtype), str(t.device))
                stats[key] = stats.get(key, 0) + 1
            except Exception:
                pass
        return stats

    stats_before = get_stats(before)
    stats_after = get_stats(after)
    
    print(f"\n--- Tensor Diff: {label} ---")
    all_keys = set(stats_before.keys()) | set(stats_after.keys())
    
    has_diff = False
    for k in sorted(all_keys, key=lambda x: str(x)):
        cnt_b = stats_before.get(k, 0)
        cnt_a = stats_after.get(k, 0)
        if cnt_a != cnt_b:
            diff = cnt_a - cnt_b
            shape, dtype, device = k
            size_mb = (torch.zeros(shape, dtype=eval(f"torch.{dtype}")).numel() * torch.zeros(1, dtype=eval(f"torch.{dtype}")).element_size()) / (1024 ** 2) if "bits" not in dtype else 0.0
            print(f"  Shape: {str(shape):<20} | Dtype: {dtype:<10} | Device: {device:<10} | Diff: {diff:+d} ({cnt_b} -> {cnt_a}) | Total Diff Size: {size_mb * diff:+.3f} MB")
            has_diff = True
            
    if not has_diff:
        print("  No active tensor count difference.")

async def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    
    wrapper = DiffKVHFWrapper(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        config={"rank": 16, "micro_block_size": 16},
        device=device
    )
    
    abstract = "Sustainable AI is a movement to foster change in the entire lifecycle of AI products."
    long_text = "\n".join([f"Paragraph {i+1}:\n{abstract}" for i in range(20)])
    prompt = f"<|im_start|>user\n{long_text}\n\nWhat is it?<|im_end|>\n<|im_start|>assistant\n"
    
    engine = ContinuousBatchEngine(wrapper, max_batch_size=2)
    engine.start()
    
    # Pre-warmup GC
    gc.collect()
    torch.mps.empty_cache()
    
    print("\nSubmitting request...")
    q = await engine.submit("session_trace", {
        "prompt": prompt,
        "max_tokens": 15,
        "temperature": 0.0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
    })
    
    step = 0
    tensors_prev = get_live_tensors()
    
    while True:
        chunk = await q.get()
        if "error" in chunk:
            break
        text = chunk.get("text", "")
        if text:
            gc.collect()
            torch.mps.empty_cache()
            tensors_curr = get_live_tensors()
            print_tensor_diff(tensors_prev, tensors_curr, f"Step {step:02d} ('{text}')")
            
            # Print referrers of any tensor with shape (1, 2, *, 64)
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj) and len(obj.shape) == 4 and obj.shape[0] == 1 and obj.shape[1] == 2 and obj.shape[3] == 64:
                        L = obj.shape[2]
                        # Let's inspect tensors of size L (excluding block size 16 and anchor_kv size 2)
                        if L not in (2, 16):
                            print(f"\n[DIAGNOSTIC] Found tensor of shape {obj.shape} on {obj.device}")
                            refs = gc.get_referrers(obj)
                            print(f"  Referrers count: {len(refs)}")
                            for idx, r in enumerate(refs):
                                r_type = type(r).__name__
                                print(f"    Referrer {idx}: Type={r_type}")
                                if isinstance(r, dict):
                                    # print keys of dict
                                    keys = list(r.keys())
                                    print(f"      Dict keys (truncated): {keys[:10]}")
                                    for k in keys:
                                        if r[k] is obj:
                                            print(f"        -> key holding this tensor: {k}")
                                elif isinstance(r, list):
                                    print(f"      List length: {len(r)}")
                                elif isinstance(r, tuple):
                                    print(f"      Tuple length: {len(r)}")
                                else:
                                    try:
                                        print(f"      Repr: {str(r)[:200]}")
                                    except Exception:
                                        pass
                except Exception as e:
                    pass
            
            tensors_prev = tensors_curr
            step += 1
        if chunk.get("is_final"):
            break
            
    await engine.stop()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
