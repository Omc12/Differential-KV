"""DKV attention backend for vLLM 0.34.0.

Selected with `--attention-backend CUSTOM`. What it does depends on
DKV_VLLM_CODEC:

  unset      phase 1: every call goes to vLLM's Triton backend unchanged
             (token-identical to stock vLLM).
  k8v4 | <DKV codec>
             phases 2-3 (codec_attn.py): pages hold codec slots, sized
             through `customize_spec` the way vLLM's own quantized-KV backend
             (TurboQuant) sizes its pages; decode reads only routed pages
             when DKV_VLLM_ROUTE_TOKENS > 0 (default 0: read every page).
"""
from dataclasses import replace

from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

from .codec_attn import DKVCodecAttentionImpl, codec_name

MARKER = "[dkv_vllm] DKV attention backend active"


class DKVAttentionImpl(TritonAttentionImpl):
    """Phase 1: identical to TritonAttentionImpl."""

    _announced = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not DKVAttentionImpl._announced:       # once per process, any layer
            DKVAttentionImpl._announced = True
            print(MARKER, flush=True)


class DKVAttentionMetadataBuilder(TritonAttentionMetadataBuilder):
    """Phase 1: identical to TritonAttentionMetadataBuilder."""


class DKVCodecMetadataBuilder(TritonAttentionMetadataBuilder):
    """Codec pages. Decode-only batches take the fused, sync-free path, which
    is captured in full CUDA graphs; prefill and mixed batches run the
    reference path eagerly (it loops over requests in Python)."""
    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE


class DKVAttentionBackend(TritonAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @classmethod
    def customize_spec(cls, spec):
        spec = super().customize_spec(spec)
        name = codec_name()
        if not name or spec.state_content_bytes is not None:
            return spec
        from .slot_codecs import make_slot_codec
        c = make_slot_codec(name, spec.num_kv_heads, spec.head_size)
        return replace(spec, state_content_bytes=c.slot_bytes)

    @classmethod
    def supported_kv_cache_layouts(cls):
        if codec_name():
            return (KVCacheLayout.LBHNC,)     # per layer: [blocks, H, tokens, C]
        return super().supported_kv_cache_layouts()

    @staticmethod
    def get_impl_cls():
        return DKVCodecAttentionImpl if codec_name() else DKVAttentionImpl

    @staticmethod
    def get_builder_cls():
        return DKVCodecMetadataBuilder if codec_name() else DKVAttentionMetadataBuilder
