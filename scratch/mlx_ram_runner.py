import os, sys, time
_d=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_d,"ACTIVE_RUNTIME"))
from serving.mlx_diffkv_wrapper import MLXDiffKVWrapper
prompt=open(sys.argv[1]).read()
w=MLXDiffKVWrapper(model_id="Qwen/Qwen2.5-1.5B-Instruct", config={"preset":"low","rank":16,"micro_block_size":256})
print("__READY__", flush=True)
msgs=[{"role":"user","content":prompt}]
pf=w.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
w.generate(prompt=pf, max_new_tokens=40, temperature=0.1, top_p=0.9, repetition_penalty=1.15)
print("__FINISH__", flush=True)
