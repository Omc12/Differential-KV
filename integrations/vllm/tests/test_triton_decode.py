"""The fused k8v4 decode kernel equals a plain reference (needs CUDA + Triton)."""
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _reference(q, pages, plist, logical, seq, scale, codec):
    R, Hq, D = q.shape
    _B, Hk, BS, _C = pages.shape
    g = Hq // Hk
    out = torch.empty_like(q, dtype=torch.float32)
    for r in range(R):
        keep = [(int(p), int(l)) for p, l in zip(plist[r].tolist(), logical[r].tolist()) if p >= 0]
        slots = pages[[p for p, _ in keep]].permute(0, 2, 1, 3).reshape(-1, Hk, pages.shape[-1])
        k, v = codec.decode(slots, torch.float32)
        pos = torch.tensor([l * BS + t for _, l in keep for t in range(BS)], device=q.device)
        ok = pos < int(seq[r])
        k, v = k[ok], v[ok]
        kh = k.transpose(0, 1).repeat_interleave(g, 0)
        vh = v.transpose(0, 1).repeat_interleave(g, 0)
        att = torch.softmax((q[r].float().unsqueeze(1) @ kh.transpose(1, 2)).squeeze(1) * scale, -1)
        out[r] = (att.unsqueeze(1) @ vh).squeeze(1)
    return out


@pytest.mark.parametrize("D,Hk", [(128, 2), (64, 8)])
def test_store_kernel_writes_the_reference_bytes_and_bounds(D, Hk):
    from dkv_vllm.slot_codecs import K8V4
    from dkv_vllm.triton_decode import k8v4_store
    torch.manual_seed(1)
    dev, BS, B, N = "cuda", 16, 10, 37
    codec = K8V4(D)
    pages = torch.zeros(B, Hk, BS, codec.slot_bytes, dtype=torch.uint8, device=dev)
    kmin = torch.full((B + 1, Hk, D), float("inf"), device=dev)
    kmax = torch.full((B + 1, Hk, D), float("-inf"), device=dev)
    slots = torch.randperm(B * BS, device=dev)[:N].to(torch.int32)
    slots[5] = -1                                          # padding is skipped
    k = torch.randn(N, Hk, D, device=dev, dtype=torch.float16) * 2
    v = torch.randn(N, Hk, D, device=dev, dtype=torch.float16)
    k8v4_store(k, v, slots, pages, kmin, kmax)
    ok = slots >= 0
    s = slots[ok].long()
    ref = codec.encode(k[ok], v[ok])
    got = pages[s // BS, :, s % BS]
    # Bit-equal except where a value sits exactly on a rounding tie (seen: 1
    # nibble in 32k bytes); there both encodings are valid and decode within
    # one quantization step of each other.
    assert (got != ref).float().mean().item() < 1e-3
    kg, vg = codec.decode(got, torch.float32)
    kr, vr = codec.decode(ref, torch.float32)
    k_step = k[ok].float().abs().amax(-1, keepdim=True) / 127
    v_step = (v[ok].float().view(-1, Hk, D // 32, 32).amax(-1) -
              v[ok].float().view(-1, Hk, D // 32, 32).amin(-1)) / 15
    assert ((kg - kr).abs() <= k_step * 1.01 + 1e-6).all()
    assert ((vg - vr).abs().view(-1, Hk, D // 32, 32) <= v_step.unsqueeze(-1) * 1.01 + 1e-3).all()
    for p in torch.unique(s // BS).tolist():
        rows = k[ok][(s // BS) == p].float()
        assert torch.equal(kmin[p], rows.amin(0)) and torch.equal(kmax[p], rows.amax(0))


@pytest.mark.parametrize("D,Hq,Hk", [(128, 12, 2), (64, 8, 8), (128, 32, 8)])
def test_kernel_matches_reference(D, Hq, Hk):
    from dkv_vllm.slot_codecs import K8V4
    from dkv_vllm.triton_decode import k8v4_decode
    torch.manual_seed(0)
    dev, BS, B = "cuda", 16, 40
    codec = K8V4(D)
    k = torch.randn(B * BS, Hk, D, device=dev)
    v = torch.randn(B * BS, Hk, D, device=dev)
    pages = codec.encode(k, v).view(B, BS, Hk, -1).permute(0, 2, 1, 3).contiguous()
    R = 3
    seq = torch.tensor([7 * BS - 5, 3 * BS, 10 * BS - 1], device=dev, dtype=torch.int32)
    P = 12
    plist = torch.full((R, P), -1, device=dev, dtype=torch.int32)
    logical = torch.full((R, P), -1, device=dev, dtype=torch.int32)
    phys = torch.randperm(B, device=dev)
    for r, (n, routed) in enumerate([(7, None), (3, None), (10, [0, 2, 5, 8, 9])]):
        lg = list(range(n)) if routed is None else routed
        plist[r, :len(lg)] = phys[r * 12: r * 12 + len(lg)].int() if routed is None else \
            phys[torch.tensor(lg, device=dev) + 20].int()
        logical[r, :len(lg)] = torch.tensor(lg, device=dev, dtype=torch.int32)
    q = torch.randn(R, Hq, D, device=dev, dtype=torch.float16)
    scale = D ** -0.5
    got = k8v4_decode(q, pages, plist, logical, seq, scale).float()
    ref = _reference(q, pages, plist, logical, seq, scale, codec)
    err = (got - ref).abs().max().item()
    assert err < 2e-2, err


def test_page_score_kernel_matches_the_bound():
    from dkv_vllm.triton_decode import page_scores
    torch.manual_seed(2)
    dev, R, Hq, Hk, D, B, P = "cuda", 3, 12, 2, 128, 50, 20
    q = torch.randn(R, Hq, D, device=dev, dtype=torch.float16)
    kmin = torch.randn(B, Hk, D, device=dev)
    kmax = kmin + torch.rand(B, Hk, D, device=dev)
    bt = torch.randint(0, B, (R, P), device=dev, dtype=torch.int32)
    got = page_scores(q, kmin, kmax, bt)
    qg = q.float().view(R, Hk, Hq // Hk, D)
    mn, mx = kmin[bt.long()], kmax[bt.long()]
    ref = (torch.einsum("rhgd,rphd->rphg", qg.clamp(min=0), mx)
           + torch.einsum("rhgd,rphd->rphg", qg.clamp(max=0), mn)).amax(dim=(2, 3))
    assert torch.allclose(got, ref, atol=1e-3, rtol=1e-4), (got - ref).abs().max()
