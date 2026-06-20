import os, sys, time
_d=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_d,"ACTIVE_RUNTIME"))
from serving.mlx_diffkv_wrapper import MLXDiffKVWrapper
prompt=open(os.path.join(_d,"scratch","test_natural.txt")).read()
w=MLXDiffKVWrapper(model_id="Qwen/Qwen2.5-1.5B-Instruct", config={"preset":"low","rank":16,"micro_block_size":256})
msgs=[{"role":"user","content":prompt}]
pf=w.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
t0=time.time()
r=w.generate(prompt=pf, max_new_tokens=60, temperature=0.7, top_p=0.9, repetition_penalty=1.15)
print(f"\n=== MLX generated in {time.time()-t0:.1f}s ===")
print(r)
