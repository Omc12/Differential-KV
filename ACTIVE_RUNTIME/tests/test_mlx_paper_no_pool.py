import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from serving.mlx_dkv_wrapper import MLXDKVWrapper

# Read the paper text
with open("ACTIVE_RUNTIME/nat_paper_with_needle.txt", "r") as f:
    paper_text = f.read()

# Build the prompt matching the user's exact query format
question = (
    "Use only the supplied document.\n\n"
    "What verification tag was assigned to the neighborhood attention analysis?\n\n"
    "Output only the verification tag.\n\n"
    "Do not explain your reasoning.\n\n"
    "Do not output any additional words."
)

prompt = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    + paper_text + "\n\n"
    + question + "<|im_end|>\n"
    "<|im_start|>assistant\n"
)

# Initialize MLX DKV wrapper
wrapper = MLXDKVWrapper(
    model_id="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    config={"rank": 16, "block_size": 256},
)

prompt_toks = len(wrapper.tokenizer.encode(prompt))
print(f"Prompt tokens: {prompt_toks}")

# Run generation
response = wrapper.generate(
    prompt=prompt,
    max_new_tokens=32,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.15,
)

sid = wrapper.active_session or "default"
all_ids = wrapper._session_token_ids.get(sid, [])
gen_ids = all_ids[prompt_toks:]
gen_text = wrapper.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
print(f"\nGenerated output: {gen_text!r}")
wrapper.close()
