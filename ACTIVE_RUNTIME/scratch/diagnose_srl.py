import os
import sys
import torch

# Ensure ACTIVE_RUNTIME is in path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_runtime_dir = os.path.dirname(_script_dir)
if _runtime_dir not in sys.path:
    sys.path.insert(0, _runtime_dir)

# Disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["DIFFKV_USE_TORCH_COMPILE"] = "0"

# FORCE DIFFKV TO ENGAGE FOR SHORT SEQUENCES (Defaults to 4096!)
os.environ["DIFFKV_ENGAGE_THRESHOLD"] = "0"

from serving.hf_diffkv_wrapper import PyTorchDiffKVHFWrapper
from native_core.mac_utils import get_best_device

def main():
    device = get_best_device()
    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading PyTorch wrapper for {MODEL} on {device}...")
    
    wrapper = PyTorchDiffKVHFWrapper(
        MODEL,
        config={
            "rank": 8,
            "micro_block_size": 32,
            "block_size": 32,
            "serving_mode": "balanced"
        },
        device=device
    )
    
    session_id = "diagnostic-session"
    wrapper.active_session = session_id
    
    # Configure streaming manager and block overrides to force compression on all blocks
    if wrapper.manager._streaming_mgr is not None:
        wrapper.manager._streaming_mgr.recency_window = 0
        wrapper.manager._streaming_mgr.short_context_threshold = 0
        wrapper.manager._streaming_mgr.protect_block_zero = False
        wrapper.manager._streaming_mgr._should_skip_compression = lambda *args, **kwargs: False
        
        from native_core.streaming_sparse_ingest import StreamingKVBlock
        StreamingKVBlock.short_context_threshold = 0
        StreamingKVBlock.protect_block_zero = False
        print("Configured streaming manager overrides: recency_window=0, protect_block_zero=False, skip_compression=False")
        
    content = "This is a simple short content block for testing to see if SRL can build properly."
    prompt = wrapper.tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content}
        ],
        tokenize=False,
        add_generation_prompt=True
    )
    
    print("\n--- Running generate ---")
    response = wrapper.generate(prompt, max_new_tokens=5, temperature=0.0)
    print(f"Response: {response}")
    
    # Diagnose blocks
    manager = wrapper.manager
    blocks = manager.get_streaming_blocks(session_id, 0)
    print(f"\n--- Diagnostic Info ---")
    print(f"Number of blocks in layer 0: {len(blocks)}")
    for i, b in enumerate(blocks):
        print(f"Block {i}: anchor_idx={b.anchor_idx}, state={getattr(b, 'state', 'NONE')}, pool_idx={getattr(b, 'pool_idx', 'NONE')}")
        
    token_ids = manager._session_token_ids.get(session_id)
    print(f"Token IDs length: {len(token_ids) if token_ids is not None else 'None'}")
    
    print(f"Pending CPU blocks: {getattr(manager, '_pending_cpu_blocks', 0)}")
    
    srl_state = manager.get_srl_state(session_id)
    print(f"SRL state found: {srl_state is not None}")
    
    if srl_state is None:
        # Try calling finalize_srl_index explicitly and check for errors
        print("\nCalling finalize_srl_index(session_id) explicitly...")
        try:
            manager.finalize_srl_index(session_id)
            srl_state = manager.get_srl_state(session_id)
            print(f"SRL state after explicit call: {srl_state is not None}")
        except Exception as e:
            print(f"Error during explicit call: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
