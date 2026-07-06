import sys
from transformers import AutoTokenizer

# Mirrors ACTIVE_RUNTIME benchmarks/niah_recall.py build_multi_needle_prompt: 3 distinct
# passcodes at depths 0.25 / 0.50 / 0.75 + a "list all three" question. Used to check that
# native sparse prefill routing surfaces ALL needle blocks (multi-fact), not just one.
if len(sys.argv) < 2:
    print("Usage: python3 make_multineedle_prompt.py <context_length>")
    sys.exit(1)

target_tokens = int(sys.argv[1])

tokenizer = AutoTokenizer.from_pretrained("mlx-community/Qwen2.5-1.5B-Instruct-4bit")

NEEDLE_SENTS = [
    "The first secret passcode is OMEGA-7741-DELTA.",
    "The second secret passcode is SIGMA-9923-BETA.",
    "The third secret passcode is THETA-1105-ALPHA.",
]
QUESTION = "What are the three secret passcodes? List them all in order."

filler = (
    "The history of artificial intelligence is long and complex. "
    "Early AI researchers believed that machines could simulate human reasoning. "
    "Symbolic AI dominated the field for decades, followed by neural networks. "
    "Deep learning transformed AI in the 2010s with massive datasets and GPU compute. "
)
filler_toks = tokenizer.encode(filler, add_special_tokens=False)
sents_toks = []
for s in NEEDLE_SENTS:
    sents_toks.extend(tokenizer.encode(s + "\n", add_special_tokens=False))
q_toks = tokenizer.encode(QUESTION, add_special_tokens=False)

budget = target_tokens - len(sents_toks) - len(q_toks) - 80
if budget < 100:
    budget = 100
reps = budget // len(filler_toks) + 1
allf = (filler_toks * reps)[:budget]

at1 = int(len(allf) * 0.25)
at2 = int(len(allf) * 0.50)
at3 = int(len(allf) * 0.75)
p1 = tokenizer.decode(allf[:at1])
p2 = tokenizer.decode(allf[at1:at2])
p3 = tokenizer.decode(allf[at2:at3])
p4 = tokenizer.decode(allf[at3:])

prompt = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    + p1 + "\n" + NEEDLE_SENTS[0] + "\n"
    + p2 + "\n" + NEEDLE_SENTS[1] + "\n"
    + p3 + "\n" + NEEDLE_SENTS[2] + "\n"
    + p4 + "\n\n" + QUESTION + "<|im_end|>\n"
    "<|im_start|>assistant\n"
)
sys.stdout.write(prompt.replace('\n', '\\n') + '\n')
