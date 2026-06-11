import os
import sys
sys.modules["diffkv_core"] = None
import torch
from transformers import AutoTokenizer

sys.path.insert(0, "/Users/omchimurkar1/Desktop/Differential-KV/ACTIVE_RUNTIME")
from serving.hf_diffkv_wrapper import PyTorchDiffKVHFWrapper

def main():
    os.environ["DIFFKV_ENGAGE_THRESHOLD"] = "0"
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    
    user_prompt = (
        "Random Features for Large-Scale Kernel Machines Ali Rahimi and Ben Recht Abstract To accelerate the training "
        "of kernel machines, we propose to map the input data to a randomized low-dimensional feature space and then "
        "apply existing fast linear methods. Our randomized features are designed so that the inner products of the "
        "transformed data are approximately equal to those in the feature space of a user specified shift-invariant kernel. "
        "We explore two sets of random features, provide convergence bounds on their ability to approximate various radial "
        "basis kernels, and show that in large-scale classification and regression tasks linear machine learning algorithms "
        "that use these features outperform state-of-the-art large-scale kernel machines."
    )
    
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        + user_prompt + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    print("Loading PyTorch DiffKV wrapper...")
    wrapper = PyTorchDiffKVHFWrapper(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        config={"rank": 32, "micro_block_size": 63},
        device=device,
    )
    
    # Let's inspect the first layer's forward method
    first_layer_attn = wrapper.model.model.layers[0].self_attn
    orig_forward = first_layer_attn.forward
    
    def debug_forward(self, hidden_states, *args, **kwargs):
        print(f"\n[Layer 0 Forward Called] q_len={hidden_states.shape[1]}")
        # print kwargs keys
        print("  kwargs keys:", list(kwargs.keys()))
        res = orig_forward(hidden_states, *args, **kwargs)
        return res
        
    first_layer_attn.forward = debug_forward.__get__(first_layer_attn, first_layer_attn.__class__)

    # We will hook capture_prefill_kv on the manager
    original_capture = wrapper.manager.capture_prefill_kv
    def debug_capture(session_id, layer_idx, K, V):
        if layer_idx == 0:
            print(f"\n[ACTIVE_RUNTIME capture_prefill_kv] Layer {layer_idx}")
            print(f"  K shape: {K.shape}, V shape: {V.shape}")
            chunk_len = K.shape[2]
            print("  First 10 tokens of K (layer 0, head 0):")
            for t in range(min(10, chunk_len)):
                k_vec = K[0, 0, t, :10].tolist()
                print(f"    t={t}: " + " ".join(f"{v:.6f}" for v in k_vec))
            print("  First 10 tokens of V (layer 0, head 0):")
            for t in range(min(10, chunk_len)):
                v_vec = V[0, 0, t, :10].tolist()
                print(f"    t={t}: " + " ".join(f"{v:.6f}" for v in v_vec))
        original_capture(session_id, layer_idx, K, V)
        
    wrapper.manager.capture_prefill_kv = debug_capture

    session_id = "default"
    inputs = wrapper.tokenizer(prompt, return_tensors='pt').to(wrapper.device)
    prompt_ids = inputs.input_ids[0].tolist()
    
    print(f"Prompt length in tokens: {len(prompt_ids)}")
    print(f"First 10 token IDs: {prompt_ids[:10]}")
    
    wrapper.manager.clear_session(session_id)
    wrapper._session_token_ids = {session_id: []}
    
    input_ids = torch.tensor([prompt_ids], device=wrapper.device)
    prefill_len = input_ids.shape[1]
    
    wrapper.manager.init_session(session_id, prefill_len=prefill_len)
    wrapper.manager.register_prefill_tokens(session_id, torch.tensor(prompt_ids, dtype=torch.long))
    wrapper.model._diffkv_session_ids = [session_id]
    
    # Process prompt using wrapper.generate to run it properly
    print("Running wrapper.generate...")
    out = wrapper.generate(prompt, max_new_tokens=5)
    print("Output:", out)
    
    wrapper.stop()

if __name__ == "__main__":
    main()
