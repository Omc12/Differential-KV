"""Differential-KV for vLLM.

Installed with `pip install -e integrations/vllm`, this registers DKV's
attention backend with vLLM through the `vllm.general_plugins` entry point;
select it with `--attention-backend CUSTOM` (or
`LLM(..., attention_backend="CUSTOM")`).

Phase 1 (this release): the backend delegates every call to vLLM's own Triton
backend, so output is token-identical to stock vLLM. It proves the plumbing
that the codec pages (phase 2) and routed decode (phase 3) are built on; see
docs/vllm_integration.md.
"""

BACKEND_PATH = "dkv_vllm.backend.DKVAttentionBackend"
PINNED_VLLM = "0.34.0"


def register() -> None:
    """vllm.general_plugins entry point: called in every vLLM process."""
    import vllm
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

    if vllm.__version__ != PINNED_VLLM:
        # The backend interfaces change between releases without notice.
        import warnings
        warnings.warn(f"dkv_vllm is checked against vLLM {PINNED_VLLM}; found "
                      f"{vllm.__version__}. The CUSTOM backend may not load.")
    register_backend(AttentionBackendEnum.CUSTOM, BACKEND_PATH)
