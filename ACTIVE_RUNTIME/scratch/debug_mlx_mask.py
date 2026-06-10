import os
import sys
import mlx.core as mx

# Ensure root path is in sys.path
_runtime_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _runtime_dir not in sys.path:
    sys.path.insert(0, _runtime_dir)

from mlx_lm.utils import load as mlx_load

def main():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading model {model_id}...")
    model, tokenizer = mlx_load(model_id)
    
    # Let's inspect the first self_attn module's original call signature or call it
    attn = model.model.layers[0].self_attn
    print("Attention type:", type(attn))
    
    # We will temporarily monkeypatch it to inspect mask
    orig_call = type(attn).__call__
    
    def debug_call(self, x, mask=None, cache=None):
        print(f"\n[DEBUG_CALL] x shape: {x.shape}")
        print(f"[DEBUG_CALL] mask type: {type(mask)}")
        if mask is not None:
            if isinstance(mask, mx.array):
                print(f"[DEBUG_CALL] mask shape: {mask.shape}, dtype: {mask.dtype}")
                # Print the first few values of mask if tiny
                if mask.size < 100:
                    print(mask)
            else:
                print(f"[DEBUG_CALL] mask value: {mask}")
        if cache is not None:
            print(f"[DEBUG_CALL] cache type: {type(cache)}")
        return orig_call(self, x, mask=mask, cache=cache)
        
    type(attn).__call__ = debug_call
    
    # Run a forward pass
    prompt = "Apple Silicon unified memory is an architecture where"
    input_ids = mx.array([tokenizer.encode(prompt)])
    print("Input shape:", input_ids.shape)
    
    # Run
    logits = model(input_ids)
    mx.eval(logits)
    print("Logits shape:", logits.shape)

if __name__ == "__main__":
    main()
