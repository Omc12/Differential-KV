import sys
from transformers import AutoTokenizer

if len(sys.argv) < 5:
    print("Usage: python3 make_niah_prompt.py <context_length> <depth> <needle> <question>")
    sys.exit(1)

target_tokens = int(sys.argv[1])
depth = float(sys.argv[2])
needle = sys.argv[3]
question = sys.argv[4]

tokenizer = AutoTokenizer.from_pretrained("mlx-community/Qwen2.5-1.5B-Instruct-4bit")

filler = (
    "The history of artificial intelligence is long and complex. "
    "Early AI researchers believed that machines could simulate human reasoning. "
    "Symbolic AI dominated the field for decades, followed by neural networks. "
    "Deep learning transformed AI in the twentieth century with massive datasets and GPU compute. "
)

filler_toks = tokenizer.encode(filler, add_special_tokens=False)
needle_toks = tokenizer.encode(needle + "\n", add_special_tokens=False)
question_toks = tokenizer.encode(question, add_special_tokens=False)
system_overhead = 80  # approximate template tokens

filler_budget = target_tokens - len(needle_toks) - len(question_toks) - system_overhead
if filler_budget < 0:
    filler_budget = 100

repeats = (filler_budget // len(filler_toks)) + 1
all_filler = (filler_toks * repeats)[:filler_budget]

insert_at = int(len(all_filler) * depth)
part1 = tokenizer.decode(all_filler[:insert_at])
part2 = tokenizer.decode(all_filler[insert_at:])

# Construct Qwen-compatible chat prompt
prompt = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    + part1 + "\n"
    + needle + "\n"
    + part2 + "\n\n"
    + question + "<|im_end|>\n"
    "<|im_start|>assistant\n"
)

# Print escaped prompt to stdout (getline reads single line, binary unescapes \n to actual newlines)
escaped = prompt.replace('\n', '\\n')
sys.stdout.write(escaped + '\n')
