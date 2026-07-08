# ============================================================
#  DiffKV — Triton Fused Decode Kernel Benchmark
#  Self-contained Google Colab benchmark
#
#  Usage:
#    1. Open Google Colab (https://colab.research.google.com)
#    2. Runtime → Change runtime type → GPU (T4 / A100 / V100)
#    3. Upload this file OR paste each section into cells
#    4. Run all sections in order
# ============================================================

# ── SECTION 1: Install dependencies ──────────────────────────────────────────
# Uncomment and run on Colab:
# !pip install -q triton psutil tabulate

# ── SECTION 2: Verify GPU ─────────────────────────────────────────────────────
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
    print("Triton    : NOT AVAILABLE -- run: !pip install triton")
    HAS_TRITON = False
print("="*60)

if not HAS_TRITON:
    raise SystemExit("Install Triton first")

# ── SECTION 3: Kernel definitions ─────────────────────────────────────────────
from typing import Optional

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

@triton.jit
def _sparse_decode_kernel(
    q_ptr, bidx_ptr, ak_ptr, av_ptr, vk_ptr, vv_ptr,
    u_ptr, us_ptr, sc_ptr, sl_ptr,
    out_ptr, m_ptr, l_ptr,
    stride_qh, stride_qd,
    stride_akn, stride_akh, stride_akd,
    stride_avn, stride_avh, stride_avd,
    stride_vkn, stride_vkr, stride_vkh, stride_vkd,
    stride_vvn, stride_vvr, stride_vvh, stride_vvd,
    stride_un, stride_us, stride_ur,
    stride_oh, stride_od,
    N: tl.constexpr, Hq: tl.constexpr, KVG: tl.constexpr, D: tl.constexpr,
    R: tl.constexpr, S: tl.constexpr, INV: tl.constexpr,
    BPC: tl.constexpr, NC: tl.constexpr,
):
    hq = tl.program_id(0); ch = tl.program_id(1); hkv = hq // KVG
    od = tl.arange(0, D); or_ = tl.arange(0, R); os = tl.arange(0, S)
    q = tl.load(q_ptr + hq * stride_qh + od * stride_qd).to(tl.float32)
    mi = -float("inf"); li = 0.0; Oi = tl.zeros([D], dtype=tl.float32)
    sb = ch * BPC; eb = tl.minimum(sb + BPC, N)
    for n in range(sb, eb):
        pi = tl.load(bidx_ptr + n)
        sc = tl.load(sc_ptr + pi).to(tl.float32)
        as_ = tl.load(sl_ptr + pi)
        ak = tl.load(ak_ptr + pi*stride_akn + hkv*stride_akh + od*stride_akd).to(tl.float32)
        av = tl.load(av_ptr + pi*stride_avn + hkv*stride_avh + od*stride_avd).to(tl.float32)
        vk = tl.load(vk_ptr + pi*stride_vkn + hkv*stride_vkh + or_[:,None]*stride_vkr + od[None,:]*stride_vkd).to(tl.float32)
        vv = tl.load(vv_ptr + pi*stride_vvn + hkv*stride_vvh + or_[:,None]*stride_vvr + od[None,:]*stride_vvd).to(tl.float32)
        uscl = tl.load(us_ptr + pi)
        sm = os[:,None] < as_
        u = tl.load(u_ptr + pi*stride_un + os[:,None]*stride_us + or_[None,:]*stride_ur, mask=sm, other=0.0).to(tl.float32) * uscl
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
        Oi = Oi / li; tl.store(out_ptr + hq*stride_oh + od*stride_od, Oi)
        if m_ptr is not None: tl.store(m_ptr + hq, mi)
        if l_ptr is not None: tl.store(l_ptr + hq, li)
    else:
        tl.store(out_ptr + hq*(NC*D) + ch*D + od, Oi)
        if m_ptr is not None: tl.store(m_ptr + hq*NC + ch, mi)
        if l_ptr is not None: tl.store(l_ptr + hq*NC + ch, li)

@triton.jit
def _reduce_kernel(ow, mw, lw, op, mp, lp, NC: tl.constexpr, D: tl.constexpr):
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

@triton.jit
def _combined_kernel(
    q_ptr, bidx_ptr, ak_ptr, av_ptr, vk_ptr, vv_ptr,
    u_ptr, us_ptr, sc_ptr, sl_ptr,
    dk_ptr, dv_ptr, Ld,
    sdk_h, sdk_l, sdk_d, sdv_h, sdv_l, sdv_d,
    out_ptr, m_ptr, l_ptr,
    stride_qh, stride_qd,
    stride_akn, stride_akh, stride_akd,
    stride_avn, stride_avh, stride_avd,
    stride_vkn, stride_vkr, stride_vkh, stride_vkd,
    stride_vvn, stride_vvr, stride_vvh, stride_vvd,
    stride_un, stride_us, stride_ur,
    stride_oh, stride_od,
    N: tl.constexpr, Hq: tl.constexpr, KVG: tl.constexpr, D: tl.constexpr,
    R: tl.constexpr, S: tl.constexpr, INV: tl.constexpr,
    BPC: tl.constexpr, NC: tl.constexpr, DPC: tl.constexpr,
):
    hq = tl.program_id(0); ch = tl.program_id(1); hkv = hq // KVG
    od = tl.arange(0, D); or_ = tl.arange(0, R); os = tl.arange(0, S)
    q = tl.load(q_ptr + hq * stride_qh + od * stride_qd).to(tl.float32)
    mi = -float("inf"); li = 0.0; Oi = tl.zeros([D], dtype=tl.float32)
    # sparse blocks
    sb = ch * BPC; eb = tl.minimum(sb + BPC, N)
    for n in range(sb, eb):
        pi = tl.load(bidx_ptr + n)
        sc = tl.load(sc_ptr + pi).to(tl.float32)
        as_ = tl.load(sl_ptr + pi)
        ak = tl.load(ak_ptr + pi*stride_akn + hkv*stride_akh + od*stride_akd).to(tl.float32)
        av = tl.load(av_ptr + pi*stride_avn + hkv*stride_avh + od*stride_avd).to(tl.float32)
        vk = tl.load(vk_ptr + pi*stride_vkn + hkv*stride_vkh + or_[:,None]*stride_vkr + od[None,:]*stride_vkd).to(tl.float32)
        vv = tl.load(vv_ptr + pi*stride_vvn + hkv*stride_vvh + or_[:,None]*stride_vvr + od[None,:]*stride_vvd).to(tl.float32)
        uscl = tl.load(us_ptr + pi)
        sm = os[:,None] < as_
        u = tl.load(u_ptr + pi*stride_un + os[:,None]*stride_us + or_[None,:]*stride_ur, mask=sm, other=0.0).to(tl.float32) * uscl
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
    # dense window — fused in same accumulator
    if DPC > 0:
        ds_ = ch * DPC; de_ = tl.minimum(ds_ + DPC, Ld)
        for t in range(ds_, de_):
            dk = tl.load(dk_ptr + hkv*sdk_h + t*sdk_l + od*sdk_d).to(tl.float32)
            dv = tl.load(dv_ptr + hkv*sdv_h + t*sdv_l + od*sdv_d).to(tl.float32)
            sc2 = tl.sum(q * dk) * INV
            mn = tl.maximum(mi, sc2); al = tl.exp(mi-mn); p = tl.exp(sc2-mn)
            li = li*al + p; Oi = Oi*al + p*dv; mi = mn
    # epilogue
    if NC == 1:
        Oi = Oi / li; tl.store(out_ptr + hq*stride_oh + od*stride_od, Oi)
        if m_ptr is not None: tl.store(m_ptr + hq, mi)
        if l_ptr is not None: tl.store(l_ptr + hq, li)
    else:
        tl.store(out_ptr + hq*(NC*D) + ch*D + od, Oi)
        if m_ptr is not None: tl.store(m_ptr + hq*NC + ch, mi)
        if l_ptr is not None: tl.store(l_ptr + hq*NC + ch, li)

print("Kernels ready")

# ── SECTION 4: Pool + dispatch helpers ────────────────────────────────────────
class Pool:
    def __init__(self, N, S, R, Hkv, D, device, dtype=torch.float32):
        self.U  = torch.randn(N, S, R, device=device, dtype=dtype) * 0.02
        self.Us = torch.ones(N, device=device, dtype=dtype)
        self.VK = torch.randn(N, R, Hkv, D, device=device, dtype=dtype) * 0.02
        self.VV = torch.randn(N, R, Hkv, D, device=device, dtype=dtype) * 0.02
        self.aK = torch.randn(N, Hkv, D, device=device, dtype=dtype) * 0.02
        self.aV = torch.randn(N, Hkv, D, device=device, dtype=dtype) * 0.02
        self.sl = torch.full((N,), S, device=device, dtype=torch.int32)
        self.sc = torch.ones(N, device=device, dtype=dtype)

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

def sparse_only(q, bidx, pool, inv, BPC=16):
    Hq,D = q.shape; N=bidx.shape[0]
    Dp=triton.next_power_of_2(D); Rp=triton.next_power_of_2(pool.U.shape[2]); Sp=triton.next_power_of_2(pool.U.shape[1])
    nc=max(1,(N+BPC-1)//BPC); KVG=Hq//pool.aK.shape[1]
    ow,mw,lw,o,m,l = _alloc(Hq,nc,D,Dp,q.device)
    _sparse_decode_kernel[(Hq,nc)](
        q,bidx,pool.aK,pool.aV,pool.VK,pool.VV,pool.U,pool.Us,pool.sc,pool.sl,
        ow,mw,lw,
        q.stride(0),q.stride(1),
        pool.aK.stride(0),pool.aK.stride(1),pool.aK.stride(2),
        pool.aV.stride(0),pool.aV.stride(1),pool.aV.stride(2),
        pool.VK.stride(0),pool.VK.stride(1),pool.VK.stride(2),pool.VK.stride(3),
        pool.VV.stride(0),pool.VV.stride(1),pool.VV.stride(2),pool.VV.stride(3),
        pool.U.stride(0),pool.U.stride(1),pool.U.stride(2),
        o.stride(0),o.stride(1),
        N,Hq,KVG,Dp,Rp,Sp,inv,BPC,nc)
    if nc>1: _reduce_kernel[(Hq,)](ow,mw,lw,o,m,l,nc,Dp)
    return o,m,l

def combined(q, bidx, pool, dk, dv, inv, BPC=16):
    Hq,D=q.shape; N=bidx.shape[0]; L=dk.shape[1]
    Dp=triton.next_power_of_2(D); Rp=triton.next_power_of_2(pool.U.shape[2]); Sp=triton.next_power_of_2(pool.U.shape[1])
    ncs=max(1,(N+BPC-1)//BPC); KVG=Hq//pool.aK.shape[1]
    DPC=max(1,(L+ncs-1)//ncs) if L>0 else 0
    nc=max(ncs,(L+DPC-1)//DPC) if L>0 else ncs
    ow,mw,lw,o,m,l = _alloc(Hq,nc,D,Dp,q.device)
    _combined_kernel[(Hq,nc)](
        q,bidx,pool.aK,pool.aV,pool.VK,pool.VV,pool.U,pool.Us,pool.sc,pool.sl,
        dk,dv,L,
        dk.stride(0),dk.stride(1),dk.stride(2),
        dv.stride(0),dv.stride(1),dv.stride(2),
        ow,mw,lw,
        q.stride(0),q.stride(1),
        pool.aK.stride(0),pool.aK.stride(1),pool.aK.stride(2),
        pool.aV.stride(0),pool.aV.stride(1),pool.aV.stride(2),
        pool.VK.stride(0),pool.VK.stride(1),pool.VK.stride(2),pool.VK.stride(3),
        pool.VV.stride(0),pool.VV.stride(1),pool.VV.stride(2),pool.VV.stride(3),
        pool.U.stride(0),pool.U.stride(1),pool.U.stride(2),
        o.stride(0),o.stride(1),
        N,Hq,KVG,Dp,Rp,Sp,inv,BPC,nc,DPC=DPC)
    if nc>1: _reduce_kernel[(Hq,)](ow,mw,lw,o,m,l,nc,Dp)
    return o,m,l

def mem():
    ga = torch.cuda.memory_allocated()  / 1e6 if torch.cuda.is_available() else 0
    gr = torch.cuda.memory_reserved()   / 1e6 if torch.cuda.is_available() else 0
    cr = psutil.Process(os.getpid()).memory_info().rss / 1e6
    return ga, gr, cr

print("Helpers ready")

# ── SECTION 5: Parity check ───────────────────────────────────────────────────
device = "cuda"
N,S,R,Hkv,Hq,L,D = 8,64,16,8,32,256,128
inv = 1.0/math.sqrt(D)

pool = Pool(N,S,R,Hkv,D,device)
bidx = torch.arange(N, device=device, dtype=torch.int32)
q    = torch.randn(Hq,D, device=device) * 0.1
dk   = torch.randn(Hkv,L,D, device=device) * 0.1
dv   = torch.randn(Hkv,L,D, device=device) * 0.1

# 3-step reference
os_,ms_,ls_ = sparse_only(q,bidx,pool,inv)
lse_sp = ms_ + torch.log(ls_.clamp(min=1e-9))
qkv = q.view(Hkv,Hq//Hkv,D)
sd  = torch.bmm(qkv, dk.permute(0,2,1)).view(Hq,L)*inv
lse_d = torch.logsumexp(sd,-1)
wd_ = torch.softmax(sd,-1).view(Hkv,Hq//Hkv,L)
od_ = torch.bmm(wd_,dv).view(Hq,D)
lmax = torch.maximum(lse_d,lse_sp)
wd2 = torch.exp(lse_d-lmax); ws2 = torch.exp(lse_sp-lmax)
ref  = (od_*wd2.unsqueeze(-1) + os_*ws2.unsqueeze(-1))/(wd2+ws2).clamp(min=1e-9).unsqueeze(-1)

oc,_,_ = combined(q,bidx,pool,dk,dv,inv)
me = (ref-oc).abs().max().item()
print(f"Parity  max_err={me:.5f}  {'PASSED' if me<0.05 else 'FAILED'}")

# ── SECTION 6: Benchmark ──────────────────────────────────────────────────────
try:
    from tabulate import tabulate
    HAS_TAB = True
except ImportError:
    HAS_TAB = False

WARMUP, REPS = 10, 200

configs = [
    (4,  128,  "4K ctx"),
    (8,  256,  "8K ctx"),
    (16, 512,  "16K ctx"),
    (32, 1024, "32K ctx"),
]
R,Hkv,Hq,S,D = 32,8,32,128,128
inv = 1.0/math.sqrt(D)

rows = []
print(f"\nGPU: {torch.cuda.get_device_name(0)}   R={R} Hkv={Hkv} Hq={Hq} S={S} D={D}")
print(f"{'Context':<10} {'3-step(ms)':>12} {'combined(ms)':>14} {'speedup':>9} {'peak VRAM':>11} {'CPU RSS':>9}")
print("-"*70)

for N, L, label in configs:
    torch.cuda.reset_peak_memory_stats(); gc.collect(); torch.cuda.empty_cache()
    pool = Pool(N,S,R,Hkv,D,device)
    bidx = torch.arange(N,device=device,dtype=torch.int32)
    q    = torch.randn(Hq,D,device=device)
    dk   = torch.randn(Hkv,L,D,device=device)
    dv   = torch.randn(Hkv,L,D,device=device)
    qkv  = q.view(Hkv,Hq//Hkv,D)

    for _ in range(WARMUP):
        sparse_only(q,bidx,pool,inv); combined(q,bidx,pool,dk,dv,inv); torch.cuda.synchronize()

    # 3-step
    t0=time.perf_counter()
    for _ in range(REPS):
        os_,ms_,ls_ = sparse_only(q,bidx,pool,inv)
        sd = torch.bmm(qkv,dk.permute(0,2,1)).view(Hq,L)*inv
        wd_ = torch.softmax(sd,-1).view(Hkv,Hq//Hkv,L)
        od_ = torch.bmm(wd_,dv).view(Hq,D)
        lse_d = torch.logsumexp(sd,-1); lse_sp = ms_+torch.log(ls_.clamp(min=1e-9))
        lmax = torch.maximum(lse_d,lse_sp)
        wd2=torch.exp(lse_d-lmax); ws2=torch.exp(lse_sp-lmax)
        _ = (od_*wd2.unsqueeze(-1)+os_*ws2.unsqueeze(-1))/(wd2+ws2).clamp(1e-9).unsqueeze(-1)
        torch.cuda.synchronize()
    t_ref = (time.perf_counter()-t0)/REPS*1000

    # combined
    t0=time.perf_counter()
    for _ in range(REPS):
        combined(q,bidx,pool,dk,dv,inv); torch.cuda.synchronize()
    t_cb = (time.perf_counter()-t0)/REPS*1000

    spd = t_ref/t_cb
    pv  = torch.cuda.max_memory_allocated()/1e6
    _,_,cr = mem()

    rows.append([label, f"{t_ref:.3f}", f"{t_cb:.3f}", f"{spd:.2f}x", f"{pv:.0f} MB", f"{cr:.0f} MB"])
    print(f"{label:<10} {t_ref:>12.3f} {t_cb:>14.3f} {spd:>9.2f}x {pv:>9.0f} MB {cr:>7.0f} MB")

    del pool,q,dk,dv,bidx; torch.cuda.empty_cache(); gc.collect()

# ── SECTION 7: Percentile latency ─────────────────────────────────────────────
import numpy as np
N,L,D=16,512,128; R,Hkv,Hq,S=32,8,32,128; inv=1.0/math.sqrt(D)
pool=Pool(N,S,R,Hkv,D,device); bidx=torch.arange(N,device=device,dtype=torch.int32)
q=torch.randn(Hq,D,device=device); dk=torch.randn(Hkv,L,D,device=device); dv=torch.randn(Hkv,L,D,device=device)

for _ in range(30):
    sparse_only(q,bidx,pool,inv); combined(q,bidx,pool,dk,dv,inv); torch.cuda.synchronize()

REPS2=500; ts=[]; tc=[]
for _ in range(REPS2):
    s=time.perf_counter(); sparse_only(q,bidx,pool,inv); torch.cuda.synchronize(); ts.append((time.perf_counter()-s)*1000)
for _ in range(REPS2):
    s=time.perf_counter(); combined(q,bidx,pool,dk,dv,inv); torch.cuda.synchronize(); tc.append((time.perf_counter()-s)*1000)

ts=np.array(ts); tc=np.array(tc)
print(f"\n{'Percentile':<12} {'sparse-only':>14} {'combined':>14}")
print("-"*42)
for p in [50,90,95,99]:
    print(f"  P{p:<9} {np.percentile(ts,p):>14.3f} {np.percentile(tc,p):>14.3f}")
print(f"  mean       {ts.mean():>14.3f} {tc.mean():>14.3f}  ms")
print(f"  std        {ts.std():>14.3f} {tc.std():>14.3f}  ms")
tps_s=1000/(ts.mean()*32); tps_c=1000/(tc.mean()*32)
print(f"\n  TPS (32 layers, attn only):")
print(f"    sparse-only : {tps_s:.1f}")
print(f"    combined    : {tps_c:.1f}  ({tps_c/tps_s:.2f}x vs sparse-only)")
print("\nDone.")
