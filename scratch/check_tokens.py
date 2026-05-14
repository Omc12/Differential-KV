from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
needle = "IDENTIFIER-ALPHA-9-SPARSE-SIGMA"
tokens = tokenizer.encode(needle, add_special_tokens=False)
print(f"Needle: {needle}")
print(f"Tokens: {tokens}")
for t in tokens:
    print(f"  Token {t}: '{tokenizer.decode([t])}'")
