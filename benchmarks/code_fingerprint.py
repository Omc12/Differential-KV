"""Fingerprint the DKV decode path, so a code change invalidates old rows.

WHY THIS EXISTS
---------------
The checkpoint's config guard catches a changed preset, quantization or context
budget. It cannot catch a changed KERNEL, and that is exactly what bit this run:

The attention-scale fix changed DKV's decode arithmetic (granite's softmax had
been 11.3x too hot). The run config was byte-identical before and after, so on
resume the harness happily appended post-fix rows to 81 pre-fix rows and
produced a table where gov_report had jumped 10.63 -> 28.55 while hotpotqa sat
at exactly its old 15.12. Half a result, averaged with half of a different one.
Nothing errored, and the mixture is invisible in the output.

So the fingerprint hashes the SOURCE of the files that decide what DKV computes
at decode time. It changes when the arithmetic changes and not when a comment
elsewhere in the repo does. Recorded in the run config for DKV arms only, since
the baseline arms do not execute any of this code.

This deliberately does NOT use the git commit id: that changes for a README
edit, which would throw away good GPU-hours for nothing.
"""

from __future__ import annotations

import hashlib
import os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The files that determine DKV's decode arithmetic. If what these compute
# changes, previously measured DKV rows are not comparable to new ones.
_DECODE_SOURCES = (
    "ACTIVE_RUNTIME/native_core/sparse_decode/triton_fused_decode.py",
    "ACTIVE_RUNTIME/native_core/sparse_decode/remat_cache.py",
    "ACTIVE_RUNTIME/runtime/dkv_attention.py",
    "ACTIVE_RUNTIME/native_core/config.py",
)


def decode_fingerprint(short: int = 12) -> str:
    """Stable hash of the decode-critical sources. '' if none can be read."""
    h = hashlib.sha256()
    seen = 0
    for rel in _DECODE_SOURCES:
        p = os.path.join(_REPO, rel)
        try:
            with open(p, "rb") as f:
                data = f.read()
        except OSError:
            continue
        # Normalise line endings so a CRLF checkout does not read as a change.
        data = data.replace(b"\r\n", b"\n")
        h.update(rel.encode("utf-8"))
        h.update(hashlib.sha256(data).digest())
        seen += 1
    if not seen:
        return ""
    return h.hexdigest()[:short]
