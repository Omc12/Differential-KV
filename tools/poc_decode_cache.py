"""POC — decompress-and-cache decode (HANDOFF.md §BIG-WIN). Proves materialize+SDPA reproduces
compute_decode_attention_static and is 3.6-10x faster. Run: python tools/poc_decode_cache.py
Missing (spec'd, not in POC): residual override (closes 0.936->0.99), re-route-every-N caching."""
"""POC: materialize routed blocks' K/V from low-rank form, SDPA, compare to reference DiffKV."""
import sys, os, math, time, numpy as np
sys.path.insert(0, "/Users/omchimurkar1/Desktop/Differential-KV/ACTIVE_RUNTIME")
import mlx.core as mx
from serving.mlx_diffkv_wrapper import compute_decode_attention_static, MLXKVBlockManager
sys.path.insert(0, "/Users/omchimurkar1/Desktop/Differential-KV/ACTIVE_RUNTIME/tests")
from test_diffkv_kernel_parity import exact_attention, build_diffkv_store_from_kv, cosine_sim

H_kv, H_q, D = 2, 12, 128
RANK, BLOCK_SIZE, RECENCY = 32, 256, 512
NB = 16  # blocks
S = NB * BLOCK_SIZE
rng = np.random.default_rng(0)
K = mx.array((rng.standard_normal((1, H_kv, S, D)) * 0.1).astype(np.float16))
V = mx.array((rng.standard_normal((1, H_kv, S, D)) * 0.1).astype(np.float16))
Q = mx.array((rng.standard_normal((H_q, D)) * 0.1).astype(np.float16))
scale = 1.0 / math.sqrt(D)

out_exact = exact_attention(Q, K[0], V[0], scale); mx.eval(out_exact)

mgr = MLXKVBlockManager(num_layers=1, heads=H_q, kv_heads=H_kv, head_dim=D,
                        rank=RANK, block_size=BLOCK_SIZE, recency_window=RECENCY)
mgr.max_blocks = 256; mgr.max_dense_len = RECENCY + BLOCK_SIZE
sid = "poc"; mgr.init_session(sid)
build_diffkv_store_from_kv(mgr, sid, 0, K, V)
sess = mgr.sessions[sid]; nb = sess["num_blocks"][0]
print(f"nb={nb}, dense_len={sess['dense_lens'][0]}")

# reference DiffKV output
out_ref4d = mgr.execute_decode_attention(sid, 0, Q.reshape(1, H_q, 1, D), rope=None,
                                         scale=scale, num_key_value_groups=H_q//H_kv)
out_ref = out_ref4d[0, :, 0, :]; mx.eval(out_ref)
print(f"[reference DiffKV]  cosine vs exact = {cosine_sim(out_exact, out_ref):.5f}")

# --- POC: MATERIALIZE recon K/V (anchor + U@VK, residual override) then SDPA ---
U   = sess["comp_U"][0][:nb]          # [nb, S_comp, rank]
VK  = sess["comp_VK"][0][:nb]         # [nb, kv_heads, rank, D]
VV  = sess["comp_VV"][0][:nb]
ak  = sess["comp_anc_k"][0][:nb]      # [nb, kv_heads, D]
av  = sess["comp_anc_v"][0][:nb]
sc  = sess["comp_scale"][0][:nb]      # [nb]
rk  = sess["comp_res_k"][0][:nb]      # [nb, R, kv_heads, D]
rv  = sess["comp_res_v"][0][:nb]
res_mask = sess["comp_res_mask"][0][:nb] if "comp_res_mask" in sess else None
S_comp = BLOCK_SIZE - 1

def materialize():
    # reference math: recon[t] = anchor + comp_scale*(U[t] @ V_basis); anchor added to EACH delta
    delta_k = mx.einsum("bsr,bhrd->bhsd", U, VK) * sc.reshape(nb,1,1,1)  # [nb,kv_heads,S_comp,D]
    delta_v = mx.einsum("bsr,bhrd->bhsd", U, VV) * sc.reshape(nb,1,1,1)
    ak_e = mx.expand_dims(ak, 2); av_e = mx.expand_dims(av, 2)           # [nb,kv_heads,1,D]
    recon_k = ak_e + delta_k                                            # add anchor to each delta
    recon_v = av_e + delta_v
    # block = [anchor_row, delta_rows...] -> [nb, kv_heads, block_size, D]
    full_k = mx.concatenate([ak_e, recon_k], axis=2)
    full_v = mx.concatenate([av_e, recon_v], axis=2)
    # reshape to [kv_heads, nb*block_size, D]
    mk = full_k.transpose(1,0,2,3).reshape(H_kv, nb*BLOCK_SIZE, D)
    mv = full_v.transpose(1,0,2,3).reshape(H_kv, nb*BLOCK_SIZE, D)
    return mk, mv

dl = int(sess["dense_lens"][0])
dk = sess["dense_keys"][0][0, :, :dl, :]    # [kv_heads, dl, D]
dv = sess["dense_values"][0][0, :, :dl, :]
def build_kv():
    mk, mv = materialize()
    # concat materialized compressed blocks + the exact dense recency window
    fk = mx.concatenate([mk, dk], axis=1)   # [kv_heads, nb*BS + dl, D]
    fv = mx.concatenate([mv, dv], axis=1)
    return fk, fv
fk, fv = build_kv(); mx.eval(fk, fv)
L = fk.shape[1]
qs = Q.reshape(1, H_q, 1, D)
ks = fk.reshape(1, H_kv, L, D)
vs = fv.reshape(1, H_kv, L, D)
out_poc = mx.fast.scaled_dot_product_attention(qs, ks, vs, scale=scale)[0,:,0,:]; mx.eval(out_poc)
print(f"[POC materialize+SDPA] cosine vs exact = {cosine_sim(out_exact, out_poc):.5f}")
print(f"[POC vs reference DiffKV] cosine = {cosine_sim(out_ref, out_poc):.5f}")

# --- speed: materialize+SDPA per 'token' vs the reference decode ---
def timeit(fn, iters=200):
    for _ in range(10): mx.eval(fn())
    t0=time.perf_counter()
    for _ in range(iters): mx.eval(fn())
    return iters/(time.perf_counter()-t0)
sdpa_only = timeit(lambda: mx.fast.scaled_dot_product_attention(qs, ks, vs, scale=scale))
mat_sdpa  = timeit(lambda: (lambda fk_,fv_: mx.fast.scaled_dot_product_attention(qs, fk_.reshape(1,H_kv,fk_.shape[1],D), fv_.reshape(1,H_kv,fv_.shape[1],D), scale=scale))(*build_kv()))
print(f"[speed 1-layer] SDPA-only(cached)={sdpa_only:.0f}/s  materialize+SDPA(every token)={mat_sdpa:.0f}/s")
print(f"  => 28-layer tok/s: cached-SDPA ~{sdpa_only/28:.1f}   materialize-every-token ~{mat_sdpa/28:.1f}")
