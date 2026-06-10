import mlx.core as mx
from mlx_lm import load

def inspect():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading model {model_id} via mlx_lm...")
    model, tokenizer = load(model_id)
    print("Model loaded successfully!")
    print(f"Model class: {type(model)}")
    print(f"Model modules keys: {list(model.keys())}")
    
    # Inspect first layer's self_attn
    attn = model.model.layers[0].self_attn
    print(f"Attention class: {type(attn)}")
    print(f"Attention modules keys: {list(attn.keys())}")
    
if __name__ == "__main__":
    inspect()
