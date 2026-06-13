import mlx.core as mx
from mlx_lm.utils import load as mlx_load

def main():
    model_path = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
    print("Loading model...")
    model, tokenizer = mlx_load(model_path)
    
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)
    
    inputs = mx.array([[1, 2, 3, 4]])
    logits = model(inputs, cache=cache)
    mx.eval(logits)
    
    layer_0_cache = cache[0]
    print("Before trim:")
    print("keys shape:", layer_0_cache.keys.shape)
    print("offset:", layer_0_cache.offset)
    
    # Try calling trim
    try:
        layer_0_cache.trim(2)
        print("After trim(2):")
        print("keys shape:", layer_0_cache.keys.shape if layer_0_cache.keys is not None else None)
        print("offset:", layer_0_cache.offset)
    except Exception as e:
        print("Trim failed:", e)

if __name__ == "__main__":
    main()
