import os
import sys
import time

# Add ACTIVE_RUNTIME to path
_runtime_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_runtime_dir, "ACTIVE_RUNTIME"))

from serving.mlx_diffkv_wrapper import MLXDiffKVWrapper

# Read prompt
with open(os.path.join(_runtime_dir, "diffkv_native", "scratch_longprompt.txt"), "r") as f:
    prompt = f.read()

config = {
    "preset": "low",
    "rank": 16,
    "micro_block_size": 256,
}

print("Loading MLX model...")
wrapper = MLXDiffKVWrapper(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    config=config,
)

# Format using Qwen2.5 chat template
messages = [{"role": "user", "content": prompt}]
prompt_formatted = wrapper.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print("Starting generation...")
t0 = time.time()
response = wrapper.generate(
    prompt=prompt_formatted,
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.15,
)
print(f"Generated in {time.time() - t0:.2f}s:")
print("-" * 60)
print(response)
print("-" * 60)
