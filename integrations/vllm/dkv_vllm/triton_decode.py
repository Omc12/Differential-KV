"""Fused decode attention over k8v4 codec pages (phase 3 speed path).

One program per (request, KV head): walk the request's page list (routed or
all), dequantize each page's keys (int8 x fp16 scale per token) and 4-bit
values (fp16 scale/zero per 32 channels) in registers, and run an online
softmax for every query head of the GQA group at once. No 16-bit K/V is ever
written to memory -- that, and reading only the routed pages, is the speed.

Slot layout (slot_codecs.K8V4, per head per token, C bytes):
    [k int8 : D | k scale fp16 : 2 | v nibbles : D/2 | v scale fp16 : D/32*2 |
     v zero fp16 : D/32*2]
Every fp16 field starts at an even byte, so the kernel reads them through an
fp16 view of the same buffer.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _k8v4_decode_kernel(
    Q, PM, PL, PACC, PAGES_U8, PAGES_I8, PAGES_F16,
    PLIST, LOGI, SEQ, SCALE,
    stride_qr, stride_qh,
    stride_pb, stride_ph, stride_pt,            # page strides in BYTES
    stride_lr, npages, chunk,
    G: tl.constexpr, BLOCK_G: tl.constexpr, NSPLIT: tl.constexpr,
    D: tl.constexpr, BS: tl.constexpr, GROUP: tl.constexpr,
):
    """Program (r, h, sp): pages [sp * chunk, (sp + 1) * chunk) of request r,
    KV head h. Writes that chunk's running max, sum and UNNORMALISED output;
    k8v4_decode merges the NSPLIT chunks by log-sum-exp (flash-decoding)."""
    r = tl.program_id(0)
    h = tl.program_id(1)
    sp = tl.program_id(2)
    L = tl.load(SEQ + r)
    gi = tl.arange(0, BLOCK_G)
    gmask = gi < G
    d = tl.arange(0, D)
    dh = tl.arange(0, D // 2)
    t = tl.arange(0, BS)
    qh = h * G + gi
    q = tl.load(Q + r * stride_qr + qh[:, None] * stride_qh + d[None, :],
                mask=gmask[:, None], other=0.0).to(tl.float32)          # [BG, D]
    # even / odd channel halves of q (values are packed as nibble pairs)
    m_i = tl.full([BLOCK_G], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_G], tl.float32)
    acc_e = tl.zeros([BLOCK_G, D // 2], tl.float32)
    acc_o = tl.zeros([BLOCK_G, D // 2], tl.float32)
    KS_OFF: tl.constexpr = D                     # bytes
    VQ_OFF: tl.constexpr = D + 2
    VS_OFF: tl.constexpr = D + 2 + D // 2
    VZ_OFF: tl.constexpr = VS_OFF + (D // GROUP) * 2
    i0 = sp * chunk
    i1 = tl.minimum(i0 + chunk, npages)
    for i in range(i0, i1):             # runtime bounds: no unrolling
        p = tl.load(PLIST + r * stride_lr + i)
        lp = tl.load(LOGI + r * stride_lr + i)
        if p >= 0:
            base = p * stride_pb + h * stride_ph                         # bytes
            row = base + t * stride_pt                                   # [BS]
            k = tl.load(PAGES_I8 + row[:, None] + d[None, :]).to(tl.float32)        # [BS, D]
            ks = tl.load(PAGES_F16 + (row + KS_OFF) // 2).to(tl.float32)            # [BS]
            k = k * ks[:, None]
            s = tl.dot(q, tl.trans(k)) * SCALE                                       # [BG, BS]
            pos = lp * BS + t
            s = tl.where((pos < L)[None, :] & gmask[:, None], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            alpha = tl.exp(m_i - m_safe)
            pr = tl.exp(s - m_safe[:, None])
            l_i = l_i * alpha + tl.sum(pr, 1)
            vb = tl.load(PAGES_U8 + row[:, None] + VQ_OFF + dh[None, :]).to(tl.int32)  # [BS, D/2]
            grp = dh // (GROUP // 2)                                                 # group of 2i / 2i+1
            vs = tl.load(PAGES_F16 + ((row[:, None] + VS_OFF) // 2) + grp[None, :]).to(tl.float32)
            vz = tl.load(PAGES_F16 + ((row[:, None] + VZ_OFF) // 2) + grp[None, :]).to(tl.float32)
            v_e = (vb & 0xF).to(tl.float32) * vs + vz
            v_o = (vb >> 4).to(tl.float32) * vs + vz
            prv = pr.to(tl.float16)
            acc_e = acc_e * alpha[:, None] + tl.dot(prv, v_e.to(tl.float16)).to(tl.float32)
            acc_o = acc_o * alpha[:, None] + tl.dot(prv, v_o.to(tl.float16)).to(tl.float32)
            m_i = m_new
    # partials, laid out [R, Hk, NSPLIT, BLOCK_G, (D)]
    cell = (r * tl.num_programs(1) + h) * NSPLIT + sp
    tl.store(PM + cell * BLOCK_G + gi, m_i)
    tl.store(PL + cell * BLOCK_G + gi, l_i)
    abase = PACC + (cell * BLOCK_G + gi[:, None]) * D
    tl.store(abase + 2 * dh[None, :], acc_e)
    tl.store(abase + 2 * dh[None, :] + 1, acc_o)


@triton.jit
def _k8v4_store_kernel(
    K, V, SLOT, PAGES_U8, PAGES_I8, PAGES_F16, KMIN, KMAX,
    stride_kn, stride_kh, stride_vn, stride_vh,
    stride_pb, stride_ph, stride_pt,            # page strides in BYTES
    stride_bb, stride_bh,                       # bounds strides (elements)
    D: tl.constexpr, BS: tl.constexpr, GROUP: tl.constexpr,
):
    """One program per (token, KV head): quantize, pack, write the slot, and
    fold the key into its page's min / max (atomics; fp32 bounds)."""
    n = tl.program_id(0)
    h = tl.program_id(1)
    slot = tl.load(SLOT + n)
    if slot >= 0:
        blk = slot // BS
        off = slot % BS
        d = tl.arange(0, D)
        dh = tl.arange(0, D // 2)
        k = tl.load(K + n * stride_kn + h * stride_kh + d).to(tl.float32)
        ks = tl.extra.cuda.libdevice.div_rn(tl.maximum(tl.max(tl.abs(k), 0), 1e-8), 127.0)
        # IEEE division (div_rn), not the approximate default: bit-equal to torch
        kq = tl.extra.cuda.libdevice.rint(tl.extra.cuda.libdevice.div_rn(k, ks))
        kq = tl.minimum(tl.maximum(kq, -127.0), 127.0).to(tl.int8)
        row = blk * stride_pb + h * stride_ph + off * stride_pt
        tl.store(PAGES_I8 + row + d, kq)
        tl.store(PAGES_F16 + (row + D) // 2, ks.to(tl.float16))
        # values: 4-bit asymmetric per GROUP channels, as nibble pairs
        ve = tl.load(V + n * stride_vn + h * stride_vh + 2 * dh).to(tl.float32)
        vo = tl.load(V + n * stride_vn + h * stride_vh + 2 * dh + 1).to(tl.float32)
        NG: tl.constexpr = D // GROUP
        HG: tl.constexpr = GROUP // 2
        ve2 = tl.reshape(ve, (NG, HG))
        vo2 = tl.reshape(vo, (NG, HG))
        lo = tl.minimum(tl.min(ve2, 1), tl.min(vo2, 1))              # [NG]
        hi = tl.maximum(tl.max(ve2, 1), tl.max(vo2, 1))
        vs = tl.extra.cuda.libdevice.div_rn(tl.maximum(hi - lo, 1e-8), 15.0)
        # round through fp16 exactly as the stored scale/zero will be read back
        vs16 = vs.to(tl.float16).to(tl.float32)
        lo16 = lo.to(tl.float16).to(tl.float32)
        qe = tl.extra.cuda.libdevice.rint(tl.extra.cuda.libdevice.div_rn(ve2 - lo[:, None], vs[:, None]))
        qo = tl.extra.cuda.libdevice.rint(tl.extra.cuda.libdevice.div_rn(vo2 - lo[:, None], vs[:, None]))
        qe = tl.minimum(tl.maximum(qe, 0.0), 15.0).to(tl.int32)
        qo = tl.minimum(tl.maximum(qo, 0.0), 15.0).to(tl.int32)
        packed = tl.reshape(qe | (qo << 4), (D // 2,)).to(tl.uint8)
        tl.store(PAGES_U8 + row + D + 2 + dh, packed)
        g = tl.arange(0, NG)
        tl.store(PAGES_F16 + (row + D + 2 + D // 2) // 2 + g, vs16.to(tl.float16))
        tl.store(PAGES_F16 + (row + D + 2 + D // 2 + 2 * NG) // 2 + g, lo16.to(tl.float16))
        bptr = blk * stride_bb + h * stride_bh + d
        tl.atomic_min(KMIN + bptr, k)
        tl.atomic_max(KMAX + bptr, k)


def k8v4_store(key, value, slot_mapping, pages, kmin, kmax, group=32):
    """key/value [N, Hk, D]; slot_mapping [N] (-1 = padding); pages uint8
    [B, Hk, BS, C]; kmin/kmax fp32 [B (+ spare), Hk, D] page key bounds."""
    N, Hk, D = key.shape
    _B, _H, BS, C = pages.shape
    assert pages.stride(-1) == 1 and C % 2 == 0
    _k8v4_store_kernel[(N, Hk)](
        key, value, slot_mapping, pages, pages.view(torch.int8), pages.view(torch.float16),
        kmin, kmax,
        key.stride(0), key.stride(1), value.stride(0), value.stride(1),
        pages.stride(0), pages.stride(1), pages.stride(2),
        kmin.stride(0), kmin.stride(1),
        D=D, BS=BS, GROUP=group,
    )


@triton.jit
def _page_score_kernel(
    Q, KMIN, KMAX, BT, SCORE,
    stride_qr, stride_qh, stride_bb, stride_bh, stride_btr, P, Hk,
    G: tl.constexpr, D: tl.constexpr, BLOCK_P: tl.constexpr,
):
    """score[r, p] = max over heads of sum_d max(q_d kmin_d, q_d kmax_d) for
    page block_table[r, p] -- Quest's upper bound on q.k inside the page --
    without gathering the bounds into a [R, P, H, D] tensor."""
    r = tl.program_id(0)
    pb = tl.program_id(1)
    p = pb * BLOCK_P + tl.arange(0, BLOCK_P)
    pm = p < P
    phys = tl.load(BT + r * stride_btr + p, mask=pm, other=0)
    phys = tl.maximum(phys, 0)
    d = tl.arange(0, D)
    best = tl.full([BLOCK_P], float("-inf"), tl.float32)
    for h in range(0, Hk):
        bptr = phys[:, None] * stride_bb + h * stride_bh + d[None, :]
        kmn = tl.load(KMIN + bptr, mask=pm[:, None], other=0.0)                 # [BP, D]
        kmx = tl.load(KMAX + bptr, mask=pm[:, None], other=0.0)
        for g in range(0, G):
            q = tl.load(Q + r * stride_qr + (h * G + g) * stride_qh + d).to(tl.float32)
            s = tl.sum(tl.maximum(q[None, :] * kmn, q[None, :] * kmx), 1)
            best = tl.maximum(best, s)
    tl.store(SCORE + r * P + p, best, mask=pm)


def page_scores(q, kmin, kmax, block_table):
    """[R, P] fp32 upper-bound scores (pages past a request's end: garbage,
    the caller masks them)."""
    R, Hq, D = q.shape
    Hk = kmin.shape[1]
    P = block_table.shape[1]
    score = torch.empty(R, P, device=q.device, dtype=torch.float32)
    BLOCK_P = 16
    _page_score_kernel[(R, triton.cdiv(P, BLOCK_P))](
        q, kmin, kmax, block_table, score,
        q.stride(0), q.stride(1), kmin.stride(0), kmin.stride(1), block_table.stride(0),
        P, Hk, G=Hq // Hk, D=D, BLOCK_P=BLOCK_P,
    )
    return score


def k8v4_decode(q, pages, page_list, logical, seq_lens, scale, group=32):
    """q [R, Hq, D]; pages uint8 [B, Hk, BS, C]; page_list / logical int32
    [R, P] (physical page id / its index in the sequence, -1 padded);
    seq_lens int32 [R]. Returns [R, Hq, D] in q's dtype."""
    R, Hq, D = q.shape
    B, Hk, BS, C = pages.shape
    # Strided pages are fine (no copy of the cache); only the slot bytes must
    # be contiguous, so the int8 / fp16 views share the uint8 base pointer.
    assert pages.dtype == torch.uint8 and pages.stride(-1) == 1 and C % 2 == 0
    G = Hq // Hk
    BG = max(16, triton.next_power_of_2(G))
    P = page_list.shape[1]
    # enough programs to fill the GPU even for one request with few KV heads
    nsplit = max(1, min(64, triton.cdiv(P, 8)))
    chunk = triton.cdiv(P, nsplit)
    pm = torch.empty(R, Hk, nsplit, BG, device=q.device, dtype=torch.float32)
    pl = torch.empty_like(pm)
    pacc = torch.empty(R, Hk, nsplit, BG, D, device=q.device, dtype=torch.float32)
    _k8v4_decode_kernel[(R, Hk, nsplit)](
        q, pm, pl, pacc, pages, pages.view(torch.int8), pages.view(torch.float16),
        page_list, logical, seq_lens, float(scale),
        q.stride(0), q.stride(1),
        pages.stride(0), pages.stride(1), pages.stride(2),
        page_list.stride(0), P, chunk,
        G=G, BLOCK_G=BG, NSPLIT=nsplit,
        D=D, BS=BS, GROUP=group,
    )
    # log-sum-exp merge of the chunks (empty chunks carry m = -inf, l = 0)
    m = pm.amax(2, keepdim=True)
    m = torch.where(torch.isinf(m), torch.zeros_like(m), m)
    w = torch.exp(pm - m)                                   # [R, Hk, S, BG]
    num = (pacc * w.unsqueeze(-1)).sum(2)                   # [R, Hk, BG, D]
    den = (pl * w).sum(2).clamp(min=1e-30)
    out = (num / den.unsqueeze(-1))[:, :, :G].reshape(R, Hq, D)
    return out.to(q.dtype)
