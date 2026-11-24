# ============================================================
#  Anonymized custom operator benchmark (Autotuned)
# ============================================================

import torch, os, math, time, gc, sys
import psutil

print("="*60)
print(f"PyTorch   : {torch.__version__}")
print(f"CUDA avail: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU       : {torch.cuda.get_device_name(0)}")
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

try:
    import triton
    import triton.language as tl
    print(f"Triton    : {triton.__version__}  OK")
    HAS_TRITON = True
except ImportError:
    print("Triton    : NOT AVAILABLE")
    HAS_TRITON = False
print("="*60)

if not HAS_TRITON:
    raise SystemExit("Install Triton first")

# ── SECTION 1: Autotuned Kernels ──────────────────────────────────────────────

@triton.jit
def _custom_op_a(
    p_q, p_idx, p_ak, p_av, p_vk, p_vv, p_u, p_us, p_sc, p_sl,
    p_out, p_m, p_l,
    st_qh, st_qd,
    st_akn, st_akh, st_akd,
    st_avn, st_avh, st_avd,
    st_vkn, st_vkr, st_vkh, st_vkd,
    st_vvn, st_vvr, st_vvh, st_vvd,
    st_un, st_us, st_ur,
    st_oh, st_od,
    N: tl.constexpr, Hq: tl.constexpr, KVG: tl.constexpr, D: tl.constexpr,
    R: tl.constexpr, S: tl.constexpr, INV: tl.constexpr,
    BPC: tl.constexpr, NC: tl.constexpr,
):
    hq = tl.program_id(0); ch = tl.program_id(1); hkv = hq // KVG
    od = tl.arange(0, D); or_ = tl.arange(0, R); os = tl.arange(0, S)
    q = tl.load(p_q + hq * st_qh + od * st_qd).to(tl.float32)
    mi = -float("inf"); li = 0.0; Oi = tl.zeros([D], dtype=tl.float32)
    sb = ch * BPC; eb = tl.minimum(sb + BPC, N)
    for n in range(sb, eb):
        pi = tl.load(p_idx + n)
        sc = tl.load(p_sc + pi).to(tl.float32)
        as_ = tl.load(p_sl + pi)
        ak = tl.load(p_ak + pi*st_akn + hkv*st_akh + od*st_akd).to(tl.float32)
        av = tl.load(p_av + pi*st_avn + hkv*st_avh + od*st_avd).to(tl.float32)
        vk = tl.load(p_vk + pi*st_vkn + hkv*st_vkh + or_[:,None]*st_vkr + od[None,:]*st_vkd).to(tl.float32)
        vv = tl.load(p_vv + pi*st_vvn + hkv*st_vvh + or_[:,None]*st_vvr + od[None,:]*st_vvd).to(tl.float32)
        uscl = tl.load(p_us + pi)
        sm = os[:,None] < as_
        u = tl.load(p_u + pi*st_un + os[:,None]*st_us + or_[None,:]*st_ur, mask=sm, other=0.0).to(tl.float32) * uscl
        sa = tl.sum(q * ak) * INV
        qp = tl.sum(q[None,:] * vk, axis=1) * INV
        ds = tl.sum(u * qp[None,:], axis=1) * sc
        s  = tl.where(os < as_, sa + ds, -float("inf"))
        mb = tl.maximum(sa, tl.max(s, axis=0))
        mn = tl.maximum(mi, mb); al = tl.exp(mi - mn)
        pa = tl.exp(sa - mn); pd = tl.where(os < as_, tl.exp(s - mn), 0.0)
        ps = tl.sum(pd, axis=0)
        li = li * al + pa + ps
        pu = tl.sum(pd[:,None] * u, axis=0)
        od_ = tl.sum(pu[:,None] * vv, axis=0) * sc
        Oi = Oi * al + (pa + ps) * av + od_; mi = mn
    if NC == 1:
        Oi = Oi / li; tl.store(p_out + hq*st_oh + od*st_od, Oi)
        if p_m is not None: tl.store(p_m + hq, mi)
        if p_l is not None: tl.store(p_l + hq, li)
    else:
        tl.store(p_out + hq*(NC*D) + ch*D + od, Oi)
        if p_m is not None: tl.store(p_m + hq*NC + ch, mi)
        if p_l is not None: tl.store(p_l + hq*NC + ch, li)


@triton.jit
def _reduce_op(ow, mw, lw, op, mp, lp, NC: tl.constexpr, D: tl.constexpr):
    hq = tl.program_id(0); od = tl.arange(0, D)
    mi = -float("inf"); li = 0.0; Oi = tl.zeros([D], dtype=tl.float32)
    for c in range(NC):
        mc = tl.load(mw + hq*NC + c); lc = tl.load(lw + hq*NC + c)
        Oc = tl.load(ow + hq*(NC*D) + c*D + od).to(tl.float32)
        mn = tl.maximum(mi, mc); al = tl.exp(mi-mn); be = tl.exp(mc-mn)
        li = li*al + lc*be; Oi = Oi*al + Oc*be; mi = mn
    Oi = Oi / li; tl.store(op + hq*D + od, Oi)
    if mp is not None: tl.store(mp + hq, mi)
    if lp is not None: tl.store(lp + hq, li)

# Decorate with Triton's auto-tuner
@triton.autotune(
    configs=[
        triton.Config({'num_warps': 4, 'num_stages': 2}),
        triton.Config({'num_warps': 4, 'num_stages': 4}),
        triton.Config({'num_warps': 8, 'num_stages': 2}),
        triton.Config({'num_warps': 8, 'num_stages': 4}),
    ],
    key=['N', 'Ld']
)
@triton.jit
def _custom_op_b(
    p_q, p_idx, p_ak, p_av, p_vk, p_vv, p_u, p_us, p_sc, p_sl,
    p_dk, p_dv, Ld,
    st_dk_h, st_dk_l, st_dk_d, st_dv_h, st_dv_l, st_dv_d,
    p_out, p_m, p_l,
    st_qh, st_qd,
    st_akn, st_akh, st_akd,
    st_avn, st_avh, st_avd,
    st_vkn, st_vkr, st_vkh, st_vkd,
    st_vvn, st_vvr, st_vvh, st_vvd,
    st_un, st_us, st_ur,
    st_oh, st_od,
    N: tl.constexpr, Hq: tl.constexpr, KVG: tl.constexpr, D: tl.constexpr,
    R: tl.constexpr, S: tl.constexpr, INV: tl.constexpr,
    BPC: tl.constexpr, NC: tl.constexpr, DPC: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr = 64,  # Hardcoded optimal tile
):
    hq = tl.program_id(0); ch = tl.program_id(1); hkv = hq // KVG
    od = tl.arange(0, D); or_ = tl.arange(0, R); os = tl.arange(0, S)
    q = tl.load(p_q + hq * st_qh + od * st_qd).to(tl.float32)
    mi = -float("inf"); li = 0.0; Oi = tl.zeros([D], dtype=tl.float32)
    
    # Block loop
    sb = ch * BPC; eb = tl.minimum(sb + BPC, N)
    for n in range(sb, eb):
        pi = tl.load(p_idx + n)
        sc = tl.load(p_sc + pi).to(tl.float32)
        as_ = tl.load(p_sl + pi)
        ak = tl.load(p_ak + pi*st_akn + hkv*st_akh + od*st_akd).to(tl.float32)
        av = tl.load(p_av + pi*st_avn + hkv*st_avh + od*st_avd).to(tl.float32)
        vk = tl.load(p_vk + pi*st_vkn + hkv*st_vkh + or_[:,None]*st_vkr + od[None,:]*st_vkd).to(tl.float32)
        vv = tl.load(p_vv + pi*st_vvn + hkv*st_vvh + or_[:,None]*st_vvr + od[None,:]*st_vvd).to(tl.float32)
        uscl = tl.load(p_us + pi)
        sm = os[:,None] < as_
        u = tl.load(p_u + pi*st_un + os[:,None]*st_us + or_[None,:]*st_ur, mask=sm, other=0.0).to(tl.float32) * uscl
        sa = tl.sum(q * ak) * INV
        qp = tl.sum(q[None,:] * vk, axis=1) * INV
        ds = tl.sum(u * qp[None,:], axis=1) * sc
        s  = tl.where(os < as_, sa + ds, -float("inf"))
        mb = tl.maximum(sa, tl.max(s, axis=0))
        mn = tl.maximum(mi, mb); al = tl.exp(mi - mn)
        pa = tl.exp(sa - mn); pd = tl.where(os < as_, tl.exp(s - mn), 0.0)
        ps = tl.sum(pd, axis=0)
        li = li * al + pa + ps
        pu = tl.sum(pd[:,None] * u, axis=0)
        od_ = tl.sum(pu[:,None] * vv, axis=0) * sc
        Oi = Oi * al + (pa + ps) * av + od_; mi = mn

    # Extra elements loop (Vectorized/Tiled)
    if DPC > 0:
        ds_ = ch * DPC; de_ = tl.minimum(ds_ + DPC, Ld)
        for t_start in range(ds_, de_, BLOCK_SIZE_T):
            offs_t = t_start + tl.arange(0, BLOCK_SIZE_T)
            mask_t = offs_t < de_
            
            dk = tl.load(p_dk + hkv*st_dk_h + offs_t[:, None]*st_dk_l + od[None, :]*st_dk_d, mask=mask_t[:, None], other=0.0).to(tl.float32)
            scores = tl.sum(q[None, :] * dk, axis=1) * INV
            scores = tl.where(mask_t, scores, -float("inf"))
            
            mb = tl.max(scores, axis=0)
            mn = tl.maximum(mi, mb)
            al = tl.exp(mi - mn)
            p = tl.exp(scores - mn)
            p = tl.where(mask_t, p, 0.0)
            
            li = li * al + tl.sum(p, axis=0)
            
            dv = tl.load(p_dv + hkv*st_dv_h + offs_t[:, None]*st_dv_l + od[None, :]*st_dv_d, mask=mask_t[:, None], other=0.0).to(tl.float32)
            Oi = Oi * al + tl.sum(p[:, None] * dv, axis=0)
            mi = mn

    if NC == 1:
        Oi = Oi / li; tl.store(p_out + hq*st_oh + od*st_od, Oi)
        if p_m is not None: tl.store(p_m + hq, mi)
        if p_l is not None: tl.store(p_l + hq, li)
    else:
        tl.store(p_out + hq*(NC*D) + ch*D + od, Oi)
        if p_m is not None: tl.store(p_m + hq*NC + ch, mi)
        if p_l is not None: tl.store(p_l + hq*NC + ch, li)

print("Kernels ready")

# ── SECTION 2: Data structures ────────────────────────────────────────────────
class DataStore:
    def __init__(self, N, S, R, Hkv, D, device, dtype=torch.float32):
        self.u_data  = torch.randn(N, S, R, device=device, dtype=dtype) * 0.02
        self.u_scale = torch.ones(N, device=device, dtype=dtype)
        self.k_weights = torch.randn(N, R, Hkv, D, device=device, dtype=dtype) * 0.02
        self.v_weights = torch.randn(N, R, Hkv, D, device=device, dtype=dtype) * 0.02
        self.a_k = torch.randn(N, Hkv, D, device=device, dtype=dtype) * 0.02
        self.a_v = torch.randn(N, Hkv, D, device=device, dtype=dtype) * 0.02
        self.lens = torch.full((N,), S, device=device, dtype=torch.int32)
        self.scales = torch.ones(N, device=device, dtype=dtype)

def _alloc(Hq, nc, D, Dp, dev):
    if nc > 1:
        ow = torch.empty((Hq,nc,Dp), device=dev, dtype=torch.float32)
        mw = torch.empty((Hq,nc), device=dev, dtype=torch.float32)
        lw = torch.empty((Hq,nc), device=dev, dtype=torch.float32)
        o  = torch.empty((Hq,D),  device=dev, dtype=torch.float32)
        m  = torch.empty((Hq,),   device=dev, dtype=torch.float32)
        l  = torch.empty((Hq,),   device=dev, dtype=torch.float32)
    else:
        o = torch.empty((Hq,D), device=dev, dtype=torch.float32)
        m = torch.empty((Hq,),  device=dev, dtype=torch.float32)
        l = torch.empty((Hq,),  device=dev, dtype=torch.float32)
        ow,mw,lw = o,m,l
    return ow,mw,lw,o,m,l

def run_op_a(q, bidx, ds, inv, BPC=16):
    Hq,D = q.shape; N=bidx.shape[0]
    Dp=triton.next_power_of_2(D); Rp=triton.next_power_of_2(ds.u_data.shape[2]); Sp=triton.next_power_of_2(ds.u_data.shape[1])
    nc=max(1,(N+BPC-1)//BPC); KVG=Hq//ds.a_k.shape[1]
    ow,mw,lw,o,m,l = _alloc(Hq,nc,D,Dp,q.device)
    _custom_op_a[(Hq,nc)](
        q,bidx,ds.a_k,ds.a_v,ds.k_weights,ds.v_weights,ds.u_data,ds.u_scale,ds.scales,ds.lens,
        ow,mw,lw,
        q.stride(0),q.stride(1),
        ds.a_k.stride(0),ds.a_k.stride(1),ds.a_k.stride(2),
        ds.a_v.stride(0),ds.a_v.stride(1),ds.a_v.stride(2),
        ds.k_weights.stride(0),ds.k_weights.stride(1),ds.k_weights.stride(2),ds.k_weights.stride(3),
        ds.v_weights.stride(0),ds.v_weights.stride(1),ds.v_weights.stride(2),ds.v_weights.stride(3),
        ds.u_data.stride(0),ds.u_data.stride(1),ds.u_data.stride(2),
        o.stride(0),o.stride(1),
        N,Hq,KVG,Dp,Rp,Sp,inv,BPC,nc)
    if nc>1: _reduce_op[(Hq,)](ow,mw,lw,o,m,l,nc,Dp)
    return o,m,l

def run_op_b(q, bidx, ds, dk, dv, inv, BPC=16):
    Hq,D=q.shape; N=bidx.shape[0]; L=dk.shape[1]
    Dp=triton.next_power_of_2(D); Rp=triton.next_power_of_2(ds.u_data.shape[2]); Sp=triton.next_power_of_2(ds.u_data.shape[1])
    ncs=max(1,(N+BPC-1)//BPC); KVG=Hq//ds.a_k.shape[1]
    DPC=max(1,(L+ncs-1)//ncs) if L>0 else 0
    nc=max(ncs,(L+DPC-1)//DPC) if L>0 else ncs
    ow,mw,lw,o,m,l = _alloc(Hq,nc,D,Dp,q.device)
    # The autotuner handles compilation selection automatically
    _custom_op_b[(Hq,nc)](
        q,bidx,ds.a_k,ds.a_v,ds.k_weights,ds.v_weights,ds.u_data,ds.u_scale,ds.scales,ds.lens,
        dk,dv,L,
        dk.stride(0),dk.stride(1),dk.stride(2),
        dv.stride(0),dv.stride(1),dv.stride(2),
        ow,mw,lw,
        q.stride(0),q.stride(1),
        ds.a_k.stride(0),ds.a_k.stride(1),ds.a_k.stride(2),
        ds.a_v.stride(0),ds.a_v.stride(1),ds.a_v.stride(2),
        ds.k_weights.stride(0),ds.k_weights.stride(1),ds.k_weights.stride(2),ds.k_weights.stride(3),
        ds.v_weights.stride(0),ds.v_weights.stride(1),ds.v_weights.stride(2),ds.v_weights.stride(3),
        ds.u_data.stride(0),ds.u_data.stride(1),ds.u_data.stride(2),
        o.stride(0),o.stride(1),
        N,Hq,KVG,Dp,Rp,Sp,inv,BPC,nc,DPC=DPC,
        BLOCK_SIZE_T=64)
    if nc>1: _reduce_op[(Hq,)](ow,mw,lw,o,m,l,nc,Dp)
    return o,m,l

def mem():
    ga = torch.cuda.memory_allocated()  / 1e6 if torch.cuda.is_available() else 0
    gr = torch.cuda.memory_reserved()   / 1e6 if torch.cuda.is_available() else 0
    cr = psutil.Process(os.getpid()).memory_info().rss / 1e6
    return ga, gr, cr

# ── SECTION 3: Benchmark ──────────────────────────────────────────────────────
device = "cuda"
WARMUP, REPS = 20, 200

configs = [
    (4,  128,  "4K ctx"),
    (8,  256,  "8K ctx"),
    (16, 512,  "16K ctx"),
    (32, 1024, "32K ctx"),
]
R,Hkv,Hq,S,D = 32,8,32,128,128
inv = 1.0/math.sqrt(D)

print(f"\nGPU: {torch.cuda.get_device_name(0)}")
print(f"Running Autotuned Kernel Benchmarks...\n")

for N, L, label in configs:
    torch.cuda.reset_peak_memory_stats(); gc.collect(); torch.cuda.empty_cache()
    ds = DataStore(N,S,R,Hkv,D,device)
    bidx = torch.arange(N,device=device,dtype=torch.int32)
    q    = torch.randn(Hq,D,device=device)
    dk   = torch.randn(Hkv,L,D,device=device)
    dv   = torch.randn(Hkv,L,D,device=device)
    qkv  = q.view(Hkv,Hq//Hkv,D)

    # Warmup runs (Autotuner profiles during these initial calls)
    for _ in range(WARMUP):
        run_op_a(q,bidx,ds,inv)
        run_op_b(q,bidx,ds,dk,dv,inv)
        torch.cuda.synchronize()

    # 3-step baseline
    t0=time.perf_counter()
    for _ in range(REPS):
        os_,ms_,ls_ = run_op_a(q,bidx,ds,inv)
        sd = torch.bmm(qkv,dk.permute(0,2,1)).view(Hq,L)*inv
        wd_ = torch.softmax(sd,-1).view(Hkv,Hq//Hkv,L)
        od_ = torch.bmm(wd_,dv).view(Hq,D)
        lse_d = torch.logsumexp(sd,-1); lse_sp = ms_+torch.log(ls_.clamp(min=1e-9))
        lmax = torch.maximum(lse_d,lse_sp)
        wd2=torch.exp(lse_d-lmax); ws2=torch.exp(lse_sp-lmax)
        _ = (od_*wd2.unsqueeze(-1)+os_*ws2.unsqueeze(-1))/(wd2+ws2).clamp(1e-9).unsqueeze(-1)
        torch.cuda.synchronize()
    t_ref = (time.perf_counter()-t0)/REPS*1000

    # Fused autotuned
    t0=time.perf_counter()
    for _ in range(REPS):
        run_op_b(q,bidx,ds,dk,dv,inv); torch.cuda.synchronize()
    t_cb = (time.perf_counter()-t0)/REPS*1000

    spd = t_ref/t_cb
    pv  = torch.cuda.max_memory_allocated()/1e6
    _,_,cr = mem()

    # Get best configuration from the autotuner
    best_config = _custom_op_b.get_best_config() if hasattr(_custom_op_b, "get_best_config") else "default"

    print(f"{label:<10}  3-step: {t_ref:>6.3f}ms  |  Autotuned: {t_cb:>6.3f}ms  |  {spd:>5.2f}x speedup  |  (Config: {best_config})")

    del ds,q,dk,dv,bidx; torch.cuda.empty_cache(); gc.collect()

print("\nDone.")
