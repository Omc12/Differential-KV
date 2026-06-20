import os, sys, time
_d=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_d,"ACTIVE_RUNTIME"))
from serving.mlx_diffkv_wrapper import MLXDiffKVWrapper
prompt=open(os.path.join(_d,"scratch","test_bigprompt.txt")).read()
w=MLXDiffKVWrapper(model_id="Qwen/Qwen2.5-1.5B-Instruct", config={"preset":"low","rank":16,"micro_block_size":256})
msgs=[{"role":"user","content":prompt}]
pf=w.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
t0=time.time()
r=w.generate(prompt=pf, max_new_tokens=3, temperature=0.1, top_p=0.9, repetition_penalty=1.15)
print(f"[MLX] load+prefill+3tok total = {time.time()-t0:.1f}s")
