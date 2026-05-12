"""
compression/quantization_advanced.py — Phase 3 Stage F

Advanced quantization schemes for KV deltas.
Since Phase 2.5 showed quantization dominates degradation,
we test alternatives to naive INT8 symmetric.

Schemes:
  1. NF4-style — non-uniform normal-distribution-aware levels
  2. BlockWise  — per-block scales (handles outlier regions)
  3. OutlierAware — separate high-precision path for outliers
  4. AsymmetricINT8 — min/max range instead of symmetric abs-max
  5. PercentileClip — clip extremes before quantizing
"""

import math
from dataclasses import dataclass
from typing import Tuple
import torch


# ── NF4 lookup table (16 levels, normal distribution quantiles) ───────────────
_NF4_LEVELS = torch.tensor([
    -1.0000, -0.6962, -0.5251, -0.3949,
    -0.2844, -0.1848, -0.0911,  0.0000,
     0.0911,  0.1848,  0.2844,  0.3949,
     0.5251,  0.6962,  1.0000,  0.0,   # 16th level = zero
], dtype=torch.float32)


@dataclass
class QuantizedBlock:
    data: torch.Tensor    # quantized indices or values
    scale: float
    zero_point: float = 0.0
    scheme: str = "int8"

    def nbytes(self) -> int:
        bits_per_elem = {"int8": 8, "nf4": 4, "blockwise": 8, "outlier": 8}
        bpe = bits_per_elem.get(self.scheme, 8)
        return math.ceil(self.data.numel() * bpe / 8) + 8  # +8 for scale/zp

    def nbytes_vs_fp16(self) -> float:
        return self.data.numel() * 2 / (self.nbytes() + 1e-9)


# ── 1. NF4-style ─────────────────────────────────────────────────────────────

def quantize_nf4(x: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """
    NF4 quantization: map values to 16 non-uniform levels.
    Levels are spaced at normal distribution quantiles.
    Returns (int8 indices, scale).
    """
    scale = x.abs().max().item()
    if scale < 1e-9:
        return torch.zeros_like(x, dtype=torch.uint8), 1.0

    levels = _NF4_LEVELS.to(x.device)
    x_norm = x.float() / scale  # normalize to [-1, 1]

    # Nearest-neighbor assignment
    x_flat = x_norm.flatten().unsqueeze(1)  # [N, 1]
    dists   = (x_flat - levels.unsqueeze(0)).abs()  # [N, 16]
    indices = dists.argmin(dim=1).to(torch.uint8).reshape(x.shape)

    return indices, scale


def dequantize_nf4(indices: torch.Tensor, scale: float,
                   dtype: torch.dtype = torch.float16) -> torch.Tensor:
    levels = _NF4_LEVELS.to(indices.device)
    return (levels[indices.long().flatten()].reshape(indices.shape) * scale).to(dtype)


# ── 2. BlockWise INT8 ─────────────────────────────────────────────────────────

def quantize_blockwise(x: torch.Tensor,
                       block_size: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-block symmetric INT8.
    Each block has its own scale, mitigating outlier contamination.
    Returns (int8 data, scales[n_blocks]).
    """
    flat   = x.float().flatten()
    n      = flat.numel()
    n_blocks = math.ceil(n / block_size)
    padded = torch.zeros(n_blocks * block_size, device=x.device)
    padded[:n] = flat

    blocks  = padded.reshape(n_blocks, block_size)
    scales  = blocks.abs().max(dim=1).values  # [n_blocks]
    scales  = torch.where(scales < 1e-9, torch.ones_like(scales), scales)
    quant   = (blocks / scales.unsqueeze(1)).clamp(-127, 127).round().to(torch.int8)

    return quant.reshape(n_blocks, block_size), scales


def dequantize_blockwise(quant: torch.Tensor, scales: torch.Tensor,
                         original_shape: tuple,
                         dtype: torch.dtype = torch.float16) -> torch.Tensor:
    n_blocks, block_size = quant.shape
    dequant = (quant.float() * scales.unsqueeze(1)).flatten()
    n = 1
    for s in original_shape:
        n *= s
    return dequant[:n].reshape(original_shape).to(dtype)


# ── 3. Outlier-Aware ─────────────────────────────────────────────────────────

@dataclass
class OutlierQuantized:
    body_quant: torch.Tensor   # int8, non-outlier elements
    body_mask: torch.Tensor    # bool, True = non-outlier position
    body_scale: float
    outlier_vals: torch.Tensor # float16, actual outlier values
    outlier_mask: torch.Tensor # bool, True = outlier position
    shape: tuple

    def nbytes(self) -> int:
        body_bytes    = self.body_quant.numel()
        outlier_bytes = self.outlier_vals.numel() * 2
        mask_bytes    = math.ceil(self.body_mask.numel() / 8)
        return body_bytes + outlier_bytes + mask_bytes + 4


def quantize_outlier_aware(x: torch.Tensor,
                           outlier_percentile: float = 99.0) -> OutlierQuantized:
    """
    Separate outliers (top (100-p)th percentile) and quantize body as INT8.
    Outliers stored in FP16.
    """
    flat  = x.float().flatten()
    thresh = torch.quantile(flat.abs(), outlier_percentile / 100.0).item()

    outlier_mask = x.abs() > thresh
    body_mask    = ~outlier_mask

    body_vals    = x.float()[body_mask]
    body_scale   = body_vals.abs().max().item()
    if body_scale < 1e-9:
        body_scale = 1.0
    body_quant   = (body_vals / body_scale).clamp(-127, 127).round().to(torch.int8)

    outlier_vals = x[outlier_mask].to(torch.float16)

    return OutlierQuantized(
        body_quant=body_quant, body_mask=body_mask,
        body_scale=body_scale,
        outlier_vals=outlier_vals, outlier_mask=outlier_mask,
        shape=tuple(x.shape),
    )


def dequantize_outlier_aware(q: OutlierQuantized,
                              dtype: torch.dtype = torch.float16) -> torch.Tensor:
    out = torch.zeros(q.shape, dtype=torch.float32)
    out[q.body_mask]   = q.body_quant.float() * q.body_scale
    out[q.outlier_mask] = q.outlier_vals.float()
    return out.to(dtype)


# ── 4. Asymmetric INT8 ────────────────────────────────────────────────────────

def quantize_asymmetric(x: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
    """Asymmetric INT8: uses full [0, 255] range with zero-point offset."""
    x_min = x.float().min().item()
    x_max = x.float().max().item()
    scale = (x_max - x_min) / 255.0
    if scale < 1e-9:
        scale = 1.0
    zp    = -round(x_min / scale)
    quant = ((x.float() / scale) + zp).clamp(0, 255).round().to(torch.uint8)
    return quant, scale, float(zp)


def dequantize_asymmetric(quant: torch.Tensor, scale: float, zp: float,
                           dtype: torch.dtype = torch.float16) -> torch.Tensor:
    return ((quant.float() - zp) * scale).to(dtype)


# ── 5. Percentile Clip + INT8 ─────────────────────────────────────────────────

def quantize_percentile_clip(x: torch.Tensor,
                              clip_pct: float = 99.5) -> Tuple[torch.Tensor, float]:
    """Clip outliers at given percentile, then INT8 quantize."""
    flat   = x.float().flatten()
    thresh = torch.quantile(flat.abs(), clip_pct / 100.0).item()
    clipped = x.float().clamp(-thresh, thresh)
    scale   = thresh / 127.0 if thresh > 1e-9 else 1.0
    quant   = (clipped / scale).round().clamp(-127, 127).to(torch.int8)
    return quant, scale


def dequantize_percentile_clip(quant: torch.Tensor, scale: float,
                                dtype: torch.dtype = torch.float16) -> torch.Tensor:
    return (quant.float() * scale).to(dtype)


# ── Unified comparison ────────────────────────────────────────────────────────

def compare_schemes(x: torch.Tensor) -> dict:
    """
    Apply all quantization schemes to tensor x, measure reconstruction error.
    Returns dict of {scheme: {error, ratio}} for easy comparison.
    """
    results = {}

    def rms_err(recon):
        diff = (x.float() - recon.float())
        return (diff.norm() / (x.float().norm() + 1e-9)).item()

    # INT8 baseline (existing)
    from compression.quantization import quantize_int8, dequantize_int8
    q8 = quantize_int8(x)
    r8 = dequantize_int8(q8, target_dtype=torch.float32)
    results["int8_symmetric"] = {
        "error": rms_err(r8), "nbytes": q8.nbytes(),
        "ratio": x.numel() * 2 / q8.nbytes()
    }

    # NF4
    try:
        idx, sc = quantize_nf4(x)
        r_nf4   = dequantize_nf4(idx, sc, dtype=torch.float32)
        nf4_bytes = math.ceil(idx.numel() * 4 / 8) + 4
        results["nf4"] = {
            "error": rms_err(r_nf4), "nbytes": nf4_bytes,
            "ratio": x.numel() * 2 / nf4_bytes
        }
    except Exception as e:
        results["nf4"] = {"error": -1.0, "error_msg": str(e)}

    # BlockWise
    try:
        bq, bsc = quantize_blockwise(x, block_size=64)
        r_bw    = dequantize_blockwise(bq, bsc, x.shape, dtype=torch.float32)
        bw_bytes = bq.numel() + bsc.numel() * 4
        results["blockwise_int8"] = {
            "error": rms_err(r_bw), "nbytes": bw_bytes,
            "ratio": x.numel() * 2 / bw_bytes
        }
    except Exception as e:
        results["blockwise_int8"] = {"error": -1.0, "error_msg": str(e)}

    # Outlier-aware
    try:
        oq   = quantize_outlier_aware(x, outlier_percentile=99.0)
        r_oa = dequantize_outlier_aware(oq, dtype=torch.float32)
        results["outlier_aware"] = {
            "error": rms_err(r_oa), "nbytes": oq.nbytes(),
            "ratio": x.numel() * 2 / oq.nbytes()
        }
    except Exception as e:
        results["outlier_aware"] = {"error": -1.0, "error_msg": str(e)}

    # Asymmetric
    try:
        aq, asc, azp = quantize_asymmetric(x)
        r_asym = dequantize_asymmetric(aq, asc, azp, dtype=torch.float32)
        asym_bytes = aq.numel() + 8
        results["asymmetric_int8"] = {
            "error": rms_err(r_asym), "nbytes": asym_bytes,
            "ratio": x.numel() * 2 / asym_bytes
        }
    except Exception as e:
        results["asymmetric_int8"] = {"error": -1.0, "error_msg": str(e)}

    # Percentile clip
    try:
        pcq, pcs = quantize_percentile_clip(x, clip_pct=99.5)
        r_pc = dequantize_percentile_clip(pcq, pcs, dtype=torch.float32)
        pc_bytes = pcq.numel() + 4
        results["percentile_clip"] = {
            "error": rms_err(r_pc), "nbytes": pc_bytes,
            "ratio": x.numel() * 2 / pc_bytes
        }
    except Exception as e:
        results["percentile_clip"] = {"error": -1.0, "error_msg": str(e)}

    return results
