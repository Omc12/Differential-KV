"""Every quantization spelling the codebase EMITS must be one the wrapper ACCEPTS.

The bug: serving/cli.py passes `'quantization': 'int4'`, the wrapper's NF4
auto-load branch matched only the literal `"nf4"`, and nothing reconciled them.
The CLI survived because it also passes a BitsAndBytesConfig object; any caller
copying just the config dict got an UNQUANTIZED model and no message about it.

Cost of the divergence, measured on Qwen2.5-7B-Instruct at 8k on a 12 GB card:
fp16 weights are 7.62e9 x 2 = 15.2 GB, which spills to WDDM shared memory and
runs every access over PCIe -- 192.6 s/query at peak 15.49 GiB, against 7.2 s at
6.47 GiB once genuinely NF4. A 27x cliff from one unmatched string.

These are source-level assertions on purpose: they need no GPU, no bitsandbytes
and no model, so they run in ordinary CI where the failure would otherwise only
show up as an unexplained memory number in someone's benchmark.
"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ACTIVE = os.path.abspath(os.path.join(HERE, ".."))
if ACTIVE not in sys.path:
    sys.path.insert(0, ACTIVE)

from serving.hf_dkv_wrapper import QUANT_4BIT_ALIASES, QUANT_8BIT_ALIASES

ACCEPTED = set(QUANT_4BIT_ALIASES) | set(QUANT_8BIT_ALIASES)


def _string_literals(path):
    """Every string constant in a source file, via AST (no import needed)."""
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    return {n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def test_cli_emitted_spellings_are_accepted():
    # cli.py builds `'quantization': 'int4' if ... else ('int8' if ... else None)`
    for emitted in ("int4", "int8"):
        assert emitted in ACCEPTED, (
            f"cli.py emits {emitted!r} but the wrapper does not accept it; "
            f"a caller passing only the config dict loads fp16 silently"
        )


def test_preset_emitted_spelling_is_accepted():
    # native_core/config.py's `low` preset assigns the literal "nf4".
    assert "nf4" in ACCEPTED


def test_every_quantization_literal_in_cli_is_accepted():
    """Catches a FUTURE divergence, not just the one already fixed.

    Scans cli.py for the string literals assigned to a quantization key and
    asserts the wrapper knows them. Scoped to the known-emitted set rather than
    every string in the file, so it fails on a real new spelling instead of on
    unrelated prose.
    """
    cli = os.path.join(ACTIVE, "serving", "cli.py")
    literals = _string_literals(cli)
    suspects = {s for s in literals
                if s and s.lower() in {"int4", "int8", "nf4", "fp4", "4bit",
                                       "8bit", "4-bit", "8-bit", "llm.int8"}}
    assert suspects, "expected cli.py to name at least one quantization spelling"
    unknown = {s for s in suspects if s.lower() not in ACCEPTED}
    assert not unknown, (
        f"cli.py names quantization spelling(s) {sorted(unknown)} that "
        f"hf_dkv_wrapper does not accept: {sorted(ACCEPTED)}"
    )


def test_alias_sets_are_disjoint_and_lowercase():
    assert not (set(QUANT_4BIT_ALIASES) & set(QUANT_8BIT_ALIASES))
    for a in ACCEPTED:
        assert a == a.lower(), f"{a!r} must be lowercase; lookup lowercases input"
