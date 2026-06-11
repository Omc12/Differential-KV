import os
import sys

# Force DiffKV to engage regardless of prompt length
os.environ["DIFFKV_ENGAGE_THRESHOLD"] = "0"
os.environ["DIFFKV_TELEMETRY"] = "1"

import torch
import time

# Add root dir to path
sys.path.insert(0, "/Users/omchimurkar1/Desktop/Differential-KV/ACTIVE_RUNTIME")

from serving.hf_diffkv_wrapper import PyTorchDiffKVHFWrapper

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

config = {
    "rank": 32,
    "prefill_chunk_size": 512,
}

device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Initializing PyTorchDiffKVHFWrapper on {device}...")
wrapper = PyTorchDiffKVHFWrapper(
    model_id=MODEL_ID,
    config=config,
    device=device,
)

prompt = """Random Features for Large-Scale Kernel Machines Ali Rahimi and Ben Recht Abstract To accelerate the training of kernel machines, we propose to map the input data to a randomized low-dimensional feature space and then apply existing fast linear methods. Our randomized features are designed so that the inner products of the transformed data are approximately equal to those in the feature space of a user specified shift-invariant kernel. We explore two sets of random features, provide convergence bounds on their ability to approximate various radial basis kernels, and show that in large-scale classification and regression tasks linear machine learning algorithms that use these features outperform state-of-the-art large-scale kernel machines. 1 Introduction Kernel machines such as the Support Vector Machine are attractive because they can approximate any function or decision boundary arbitrarily well with enough training data. Unfortunately, methods that operate on the kernel matrix (Gram matrix) of the data scale poorly with the size of the training dataset. For example, a dataset with half a million training examples might take days to train on modern workstations. On the other hand, specialized algorithms for linear Support Vector Machines and regularized regression run much more quickly when the dimensionality of the data is small because they operate on the covariance matrix rather than the kernel matrix of the training data [1, 2]. We propose a way to combine the advantages of the linear and nonlinear approaches. Inspired by randomized algorithms for approximating kernel matrices (e.g., [3, 4]), we efficiently convert the training and evaluation of any kernel machine into the corresponding operations of a linear machine by mapping data into a relatively low-dimensional randomized feature space. Our experiments show that random features combined with very simple linear learning techniques compete favorably with state-of-the-art kernel-based classification and regression algorithms. Random features significantly reduce the computation needed for training, and obtain similar or better testing error. The kernel trick is a simple way to generate features for algorithms that depend only on the inner product between pairs of input points."""

# Wrap prompt in Qwen2.5 chat template matching C++
chat = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
chat_prompt = wrapper.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

# Hook to capture layer 0 prefill KV in PyTorch
captured_k = None
captured_v = None

def capture_hook(session_id, layer_idx, K, V):
    global captured_k, captured_v
    if layer_idx == 0 and captured_k is None:
        captured_k = K.clone().cpu()
        captured_v = V.clone().cpu()
        print(f"\n[ACTIVE_RUNTIME Hook] Captured Layer 0 Prefill KV: shape={K.shape}")

# Patch the manager to hook into capture_prefill_kv or ingest_streaming
orig_ingest = wrapper.manager.ingest_streaming
def instrumented_ingest(session_id, layer_idx, k, v):
    capture_hook(session_id, layer_idx, k, v)
    return orig_ingest(session_id, layer_idx, k, v)
wrapper.manager.ingest_streaming = instrumented_ingest

# Hook to capture routed blocks at decode step 0
decode_step_count = 0

orig_get_cached = wrapper.manager.get_cached_decode_blocks
def instrumented_get_cached(session_id, layer_idx, device):
    global decode_step_count
    res = orig_get_cached(session_id, layer_idx, device)
    if layer_idx == 0 and decode_step_count == 0:
        block_indices, dense_blocks, anchor_indices, _, _ = res
        print(f"\n[ACTIVE_RUNTIME Hook] Decode Step 0 Block Retrieval (Layer 0):")
        print(f"  block_indices (pool slots): {block_indices.tolist() if block_indices is not None else None}")
        print(f"  dense_blocks: {dense_blocks}")
        print(f"  anchor_indices: {anchor_indices.tolist() if anchor_indices is not None else None}")
        
        # Also print the actual logical block IDs matching these pool slots
        srl_state = wrapper.manager.get_srl_state(session_id)
        if srl_state is not None:
            print(f"  srl_state.ordered_slot_ids: {srl_state.ordered_slot_ids}")
            print(f"  srl_state.sink_blocks: {srl_state.sink_blocks}")
            print(f"  srl_state.current_step_slots (routed blocks): {getattr(srl_state, 'current_step_slots', None)}")
        decode_step_count += 1
    elif layer_idx == 0:
        decode_step_count += 1
    return res

wrapper.manager.get_cached_decode_blocks = instrumented_get_cached

print("Running generation...")
t0 = time.perf_counter()
response = wrapper.generate(
    prompt=chat_prompt,
    max_new_tokens=5,  # Only need first few tokens to trigger decode step 0
    temperature=0.0,
    top_p=1.0,
    repetition_penalty=1.0,
)
print(f"Generated response:\n{response}")

if captured_k is not None:
    print("\n[ACTIVE_RUNTIME] First 10 tokens of K (layer 0, head 0):")
    for t in range(min(10, captured_k.shape[2])):
        line_vals = [captured_k[0, 0, t, c].item() for c in range(10)]
        print(f"    t={t}:" + "".join(f" {v:.6f}" for v in line_vals))
        
    print("\n[ACTIVE_RUNTIME] First 10 tokens of V (layer 0, head 0):")
    for t in range(min(10, captured_v.shape[2])):
        line_vals = [captured_v[0, 0, t, c].item() for c in range(10)]
        print(f"    t={t}:" + "".join(f" {v:.6f}" for v in line_vals))

wrapper.stop()
