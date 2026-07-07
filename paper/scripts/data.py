"""Single source of measured data for every DiffKV paper figure and table.

Loads the two reconciled datasets and exposes them as plain dicts. NOTHING here
is hand-typed from a document: every number is read from a measured JSON, or
derived analytically from the model/runtime dimensions defined in the code.

Datasets (see paper/notes/AUDIT_2026-07-07.md):
  B (PRIMARY, system-level): benchmarks/results/.result_{active,dense}_{ctx}.json
     — 2026-07-04, current shipping config, active-vs-dense, 4k..64k.
  A (mechanism ablations): paper/generated/active_modes_sweep_v2.json (+_64k),
     residual_sweep.json — 2026-06-30, max_residual=64, no sparse prefill.
"""
import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
REPO = os.path.dirname(PAPER)
RESULTS = os.path.join(REPO, "benchmarks", "results")
GEN = os.path.join(PAPER, "generated")

CONTEXTS = [4096, 8192, 16384, 32768, 65536]

# ── Model / runtime dimensions (from the code; see MLXKVBlockManager.__init__
#    and Qwen2.5-1.5B config). Used for the analytic KV-byte budget. ──────────
DIMS = dict(
    model="Qwen2.5-1.5B-Instruct (int4)",
    n_layers=28,
    n_heads=12,
    kv_heads=2,          # GQA
    head_dim=128,
    block_size=256,      # S
    recency_window=512,  # dense window = recency + block = 768
    rank=32,             # r — CLI 'mid' preset default (only 'low' drops to 16)
    max_blocks=256,      # bounded pool
    fp16_bytes=2,
)


def _load(path):
    with open(path) as f:
        return json.load(f)


# ── Dataset B: primary active-vs-dense system sweep ──────────────────────────
def load_primary():
    """Return {'active': {ctx: rec}, 'dense': {ctx: rec}} from benchmarks/results.

    Prefers the freshly re-measured `fresh_{engine}_{ctx}.json` (current working
    tree, shipping config: decode-cache ON, sparse prefill ON, R=128); falls back
    to the older `.result_{engine}_{ctx}.json` only where a fresh cell is absent.
    """
    out = {"active": {}, "dense": {}}
    for engine in ("active", "dense"):
        for ctx in CONTEXTS:
            # Prefer the CLEAN rank-32 mid/balanced sweep; fall back in order.
            candidates = [
                os.path.join(RESULTS, f"clean_{engine}_{ctx}.json"),
                os.path.join(RESULTS, f"fresh_{engine}_{ctx}.json"),
                os.path.join(RESULTS, f".result_{engine}_{ctx}.json"),
            ]
            for p in candidates:
                if os.path.exists(p):
                    # First existing file (by priority) is authoritative — a clean
                    # cell that OOM'd must NOT fall back to an older successful run.
                    rec = _load(p)
                    if rec.get("status") == "ok":
                        rec["_source"] = os.path.basename(p)
                        out[engine][ctx] = rec
                    break
    return out


def primary_contexts():
    """Contexts that have BOTH active and dense measured in the primary sweep."""
    prim = load_primary()
    return [c for c in CONTEXTS if c in prim["active"] and c in prim["dense"]]


def cell_status(engine, ctx):
    """'ok' | 'oom' | 'missing' for a primary cell, reading the raw file (a failed
    cell is excluded from load_primary but its status still matters for the tables)."""
    for name in (f"clean_{engine}_{ctx}.json", f"fresh_{engine}_{ctx}.json",
                 f".result_{engine}_{ctx}.json"):
        p = os.path.join(RESULTS, name)
        if os.path.exists(p):
            return _load(p).get("status", "ok")
    return "missing"


def active_contexts():
    """Contexts where the DiffKV active engine produced a result (incl. 64k reach)."""
    prim = load_primary()
    return [c for c in CONTEXTS if c in prim["active"]]


# ── Analytic KV-state footprint for an arbitrary sequence length ─────────────
def analytic_footprint(seq_len, max_residual):
    """Whole-model DiffKV store bytes vs dense full-KV bytes at `seq_len`.

    num_blocks = full blocks that clear the recency window; the remainder stays in
    the dense window. Mirrors compress_deferred_prefill_blocks.
    """
    S = DIMS["block_size"]; W = DIMS["recency_window"]; L = DIMS["n_layers"]
    Hk = DIMS["kv_heads"]; d = DIMS["head_dim"]; b = DIMS["fp16_bytes"]
    kv_tok = Hk * d * 2 * b                     # one exact K+V token (all kv heads), one layer
    nb = max(0, (seq_len - W) // S)
    nb = min(nb, DIMS["max_blocks"])
    dense_tok = seq_len - nb * S
    bb = block_budget(max_residual)
    store = L * (nb * (bb["total"]) + dense_tok * kv_tok)
    dense = L * seq_len * kv_tok
    return dict(seq_len=seq_len, num_blocks=nb, dense_tok=dense_tok,
                store_bytes=store, dense_bytes=dense, ratio=dense / store)


# ── Dataset A: compressed-vs-exact decode ablation + analytic KV footprint ───
def load_modes():
    """Return rows list with mode in {compressed, exact}, ctx 4k..64k.

    Prefers freshly re-measured `active_modes_fresh.json`; merges any contexts
    from the older v2/_64k sweeps that the fresh run did not cover.
    """
    # Use ONLY the fresh rank-32 ablation when present (never merge older rank-16
    # runs — mixing configs across contexts is exactly the contamination to avoid).
    fresh = os.path.join(GEN, "active_modes_fresh.json")
    if os.path.exists(fresh):
        return list(_load(fresh)["results"])
    rows = []
    for name in ("active_modes_sweep_v2.json", "active_modes_sweep_64k.json"):
        p = os.path.join(GEN, name)
        if os.path.exists(p):
            rows += _load(p)["results"]
    return rows


def modes_by(mode):
    """{ctx: rec} for a given decode mode from Dataset A."""
    return {r["ctx"]: r for r in load_modes() if r.get("mode") == mode}


# ── Dataset A: residual-budget accuracy/memory sweep @16k ────────────────────
def load_residual_sweep():
    fresh = os.path.join(GEN, "residual_sweep_fresh.json")
    path = fresh if os.path.exists(fresh) else os.path.join(GEN, "residual_sweep.json")
    d = _load(path)
    rows = d.get("results", d) if isinstance(d, dict) else d
    rows = [r for r in rows if r.get("status", "ok") == "ok" and r.get("kv")]
    rows = sorted(rows, key=lambda r: r.get("max_residual", 0))
    return rows


# ── Analytic per-block KV byte budget, computed from DIMS + a residual count ─
def block_budget(max_residual):
    """Per-256-token-block bytes for the compressed store vs a dense full-KV block.

    Mirrors the live buffers in MLXKVBlockManager._create_empty_session:
      comp_U       [S-1, r]                 fp16
      comp_VK,VV   [kv_heads, r, d] each    fp16
      comp_anc_k,v [kv_heads, d] each       fp16
      comp_min_k,max_k [kv_heads, d] each   fp16
      comp_scale, comp_seq_len              ~8 B scalars
      comp_res_k,v [R, kv_heads, d] each    fp16   (R = max_residual)
    Dense block = S tokens x kv_heads x d x 2 (K and V) fp16.
    """
    S = DIMS["block_size"]; r = DIMS["rank"]; Hk = DIMS["kv_heads"]
    d = DIMS["head_dim"]; b = DIMS["fp16_bytes"]
    U   = (S - 1) * r * b
    VKV = 2 * (Hk * r * d) * b
    anc = 2 * (Hk * d) * b
    mnmx = 2 * (Hk * d) * b
    scal = 8
    lowrank = U + VKV + anc + mnmx + scal
    resid = max_residual * (2 * Hk * d) * b   # K and V exact residual tokens
    total = lowrank + resid
    dense = S * Hk * d * 2 * b
    return dict(U=U, VKV=VKV, anchors=anc, minmax=mnmx, scalars=scal,
                lowrank=lowrank, residuals=resid, total=total, dense=dense,
                ratio=dense / total, max_residual=max_residual)


if __name__ == "__main__":
    prim = load_primary()
    print("PRIMARY (Dataset B):")
    for ctx in CONTEXTS:
        a = prim["active"].get(ctx, {}); d = prim["dense"].get(ctx, {})
        print(f"  {ctx//1024:>3}k  active pf={a.get('prefill_s'):>8.2f} tps={a.get('decode_tps'):>6.2f} "
              f"mx={a.get('mx_peak_gb'):>5.2f} needle={a.get('needle_found')}   "
              f"dense pf={d.get('prefill_s'):>8.2f} tps={d.get('decode_tps'):>6.2f} mx={d.get('mx_peak_gb'):>5.2f}")
    print("\nBLOCK BUDGET  maxres=64:", {k: round(v, 3) if isinstance(v, float) else v
                                         for k, v in block_budget(64).items()})
    print("BLOCK BUDGET maxres=128:", {k: round(v, 3) if isinstance(v, float) else v
                                       for k, v in block_budget(128).items()})
