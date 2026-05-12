"""
compression/quantization_advanced.py — Phase 3 Stage F

Advanced quantization schemes for KV deltas.
Since Phase 2.5 showed quantization dominates degradation,
we test alternatives to naive INT8 symmetric.
"""

import math
from dataclasses import dataclass
from typing import Tuple, Dict
import torch

_NF4_LEVELS = torch.tensor([
    -1.0000, -0.6962, -0.5251, -0.3949,
    -0.2844, -0.1848, -0.0911,  0.0000,
    0.0911,  0.1848,  0.2844,  0.3949,
    0.5251,  0.6962,  1.0000,  0.0,
], dtype=torch.float32)

@dataclass
class QuantizedBlock:
    data: torch.Tensor
    scale: float
    zero_point: float = 0.0
    scheme: str = "int8"

    def nbytes(self) -> int:
        bits_per_elem = {"int8": 8, "nf4": 4, "blockwise": 8, "outlier": 8}
        bpe = bits_per_elem.get(self.scheme, 8)
        return math.ceil(self.data.numel() * bpe / 8) + 8

def quantize_nf4(x: torch.Tensor) -> Tuple[torch.Tensor, float]:
    scale = x.abs().max().item()
    if scale < 1e-9: return torch.zeros_like(x, dtype=torch.uint8), 1.0
    levels = _NF4_LEVELS.to(x.device)
    x_norm = x.float() / scale
    x_flat = x_norm.flatten().unsqueeze(1)
    dists = (x_flat - levels.unsqueeze(0)).abs()
    indices = dists.argmin(dim=1).to(torch.uint8).reshape(x.shape)
    return indices, scale

def dequantize_nf4(indices: torch.Tensor, scale: float, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    levels = _NF4_LEVELS.to(indices.device)
    return (levels[indices.long().flatten()].reshape(indices.shape) * scale).to(dtype)

def quantize_blockwise(x: torch.Tensor, block_size: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    flat = x.float().flatten()
    n = flat.numel()
    n_blocks = math.ceil(n / block_size)
    padded = torch.zeros(n_blocks * block_size, device=x.device)
    padded[:n] = flat
    blocks = padded.reshape(n_blocks, block_size)
    scales = blocks.abs().max(dim=1).values
    scales = torch.where(scales < 1e-9, torch.ones_like(scales), scales)
    quant = (blocks / scales.unsqueeze(1) * 127.0).clamp(-127, 127).round().to(torch.int8)
    return quant.reshape(n_blocks, block_size), scales

def dequantize_blockwise(quant: torch.Tensor, scales: torch.Tensor, original_shape: tuple, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    n_blocks, block_size = quant.shape
    dequant = (quant.float() / 127.0 * scales.unsqueeze(1)).flatten()
    n = 1
    for s in original_shape: n *= s
    return dequant[:n].reshape(original_shape).to(dtype)

@dataclass
class OutlierQuantized:
    body_quant: torch.Tensor
    body_mask: torch.Tensor
    body_scale: float
    outlier_vals: torch.Tensor
    outlier_mask: torch.Tensor
    shape: tuple
    def nbytes(self) -> int:
        return self.body_quant.numel() + self.outlier_vals.numel()*2 + math.ceil(self.body_mask.numel()/8) + 4

def quantize_outlier_aware(x: torch.Tensor, outlier_percentile: float = 99.0) -> OutlierQuantized:
    flat = x.float().flatten()
    thresh = torch.quantile(flat.abs(), outlier_percentile / 100.0).item()
    outlier_mask = x.abs() > thresh
    body_mask = ~outlier_mask
    body_vals = x.float()[body_mask]
    body_scale = body_vals.abs().max().item() if body_mask.any() else 1.0
    if body_scale < 1e-9: body_scale = 1.0
    body_quant = (body_vals / body_scale * 127.0).clamp(-127, 127).round().to(torch.int8)
    outlier_vals = x[outlier_mask].to(torch.float16)
    return OutlierQuantized(body_quant, body_mask, body_scale, outlier_vals, outlier_mask, tuple(x.shape))

def dequantize_outlier_aware(q: OutlierQuantized, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    out = torch.zeros(q.shape, dtype=torch.float32, device=q.body_quant.device)
    out[q.body_mask] = q.body_quant.float() / 127.0 * q.body_scale
    out[q.outlier_mask] = q.outlier_vals.float()
    return out.to(dtype)

def quantize_asymmetric(x: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
    x_min, x_max = x.float().min().item(), x.float().max().item()
    scale = (x_max - x_min) / 255.0 if x_max > x_min else 1.0
    zp = -round(x_min / scale)
    quant = ((x.float() / scale) + zp).clamp(0, 255).round().to(torch.uint8)
    return quant, scale, float(zp)

def dequantize_asymmetric(quant: torch.Tensor, scale: float, zp: float, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return ((quant.float() - zp) * scale).to(dtype)

def quantize_percentile_clip(x: torch.Tensor, clip_pct: float = 99.5) -> Tuple[torch.Tensor, float]:
    thresh = torch.quantile(x.float().abs(), clip_pct / 100.0).item()
    clipped = x.float().clamp(-thresh, thresh)
    scale = thresh / 127.0 if thresh > 1e-9 else 1.0
    quant = (clipped / scale).round().clamp(-127, 127).to(torch.int8)
    return quant, scale

def dequantize_percentile_clip(quant: torch.Tensor, scale: float, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return (quant.float() * scale).to(dtype)

def compare_schemes(x: torch.Tensor) -> dict:
    results = {}
    def rms_err(recon):
        return (torch.norm(x.float() - recon.float()) / (torch.norm(x.float()) + 1e-9)).item()

    from compression.quantization import quantize_int8, dequantize_int8
    q8 = quantize_int8(x)
    r8 = dequantize_int8(q8, target_dtype=torch.float32)
    results["int8_symmetric"] = {"error": rms_err(r8), "nbytes": q8.nbytes(), "ratio": x.numel()*2/q8.nbytes()}

    idx, sc = quantize_nf4(x)
    r_nf4 = dequantize_nf4(idx, sc, dtype=torch.float32)
    nb = math.ceil(idx.numel()*4/8)+4
    results["nf4"] = {"error": rms_err(r_nf4), "nbytes": nb, "ratio": x.numel()*2/nb}

    bq, bsc = quantize_blockwise(x, 64)
    r_bw = dequantize_blockwise(bq, bsc, x.shape, torch.float32)
    nb = bq.numel() + bsc.numel()*4
    results["blockwise_int8"] = {"error": rms_err(r_bw), "nbytes": nb, "ratio": x.numel()*2/nb}

    oq = quantize_outlier_aware(x, 99.0)
    r_oa = dequantize_outlier_aware(oq, torch.float32)
    results["outlier_aware"] = {"error": rms_err(r_oa), "nbytes": oq.nbytes(), "ratio": x.numel()*2/oq.nbytes()}

    aq, asc, azp = quantize_asymmetric(x)
    r_asym = dequantize_asymmetric(aq, asc, azp, torch.float32)
    nb = aq.numel() + 8
    results["asymmetric_int8"] = {"error": rms_err(r_asym), "nbytes": nb, "ratio": x.numel()*2/nb}

    pcq, pcs = quantize_percentile_clip(x, 99.5)
    r_pc = dequantize_percentile_clip(pcq, pcs, torch.float32)
    nb = pcq.numel() + 4
    results["percentile_clip"] = {"error": rms_err(r_pc), "nbytes": nb, "ratio": x.numel()*2/nb}

    return results
