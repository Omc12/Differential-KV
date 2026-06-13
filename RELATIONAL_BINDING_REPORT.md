# Relational Binding Failure — Root-Cause Report

**Date:** 2026-06-14
**Scope:** ACTIVE_RUNTIME (Python reference) + diffkv_native (C++ mirror)
**Failure class:** Concepts retrieved correctly; relationships between them reconstructed incorrectly.

---

## 1. Executive Diagnosis

The system is, architecturally, a **bag-of-spans retriever over an untyped similarity
graph.** It stores three kinds of object — entities (primes), property spans, and
*undirected, untyped, similarity-weighted edges* between spans. It never stores the
**relation itself** as a first-class object. There is nowhere in the data model that
records "EP2 —has-codimension→ 2". There are only: a span that contains the tokens
{EP2}, a span that contains the tokens {codimension, 2}, and a numeric edge weight
saying those two spans are "0.7 related."

Because the relation is never represented, it can never be **retrieved** and never be
**enforced** during generation. It is left for the language model to re-guess from token
order and from the freely-generated connective words ("is", "has", "whereas", "while").
That re-guessing is exactly the step that fails.

Every symptom in the report is a downstream consequence of this one fact:

| Observed symptom | Direct cause in this diagnosis |
|---|---|
| Properties attached to wrong concepts | No directed entity→property edge exists (RC1, RC4) |
| Correct facts combined into incorrect explanations | Relational connectives are unconstrained helpers (RC2) |
| Correct values + incorrect explanation of the value | Descriptors are surface token bags; can't bind value→meaning (RC3) |
| Distinctions between related concepts blur | Shared-vocabulary entities collapse in retrieval & entity assignment (RC3, RC4) |
| Generic explanatory filler when relationships missing | Property set truncated (top-5) → gaps filled by LM prior (RC6) |
| Comparison questions worst-affected | Entity filtering never fires on balanced 2-entity queries (RC5) |

The prior round of fixes (entity-subgraph VSL, coherence cap, sequence-start-only
fallback) all operate at the **token-gating layer**. They can reduce cross-entity *leakage*,
but none of them *create a relation representation*, so none can reconstruct a binding that
was never stored. This is why we plateaued: we have been hardening a retriever that has no
notion of "which property belongs to which entity" to begin with.

---

## 2. Root Causes (ranked by leverage)

### RC1 — Relations are not represented anywhere (highest leverage)

**Mechanism.** Both graphs are undirected and untyped.

- Factual graph edges: `weight = 0.4*sim + 0.4*lexical + 0.2*temporal`, added
  symmetrically (`factual_store.py:393-410`). The edge carries a *strength* but no
  *type* and no *direction*.
- Chunk graph edges: same shape (`chunk_graph.py:264-280`), a "handshake" weighted by
  similarity + lexical overlap + adjacency. Again untyped/undirected.

**Why it produces the failures.** A graph walk from the EP2 prime can reach the span
"{codimension, 3}" (which is EP3's property) via any edge that happens to be strong —
lexical overlap on the shared token "codimension" is enough. The walk returns it as
"related to EP2." Nothing in the structure says that span is a property *of EP3, not EP2*.
The LM then writes "EP2 has codimension 3."

**Solution.** Introduce a **typed, directed relation layer** built at prefill time:
extract `(subject_entity, relation, object/value)` triples and store directed edges
`entity --relation--> value_span`. The relation token(s) are already being preserved in
the spans (see RC2's keyword boost), so the raw material exists — it is simply discarded
into an undirected weight. Retrieval should return *triples*, not loose spans, and the
VSL should be locked to a triple's `(subject … relation … value)` ordering, not to an
arbitrary span boundary.

---

### RC2 — Relational connectives are in the HELPER set → freely hallucinated

**Mechanism.** The words that *carry* a relationship are classified as grammatical
helpers and are therefore (a) always allowed and (b) do not advance the VSL lock.

In `factual_alignment.py` the helper vocabulary (lines 16-25) includes:
`although, because, since, while, whereas, but, than, is, was, were, are, be, have, has, had`.

These are precisely the **contrastive, causal, and copular binders**. The model is free
to emit "whereas", "while", "but", "is", "has", "because" at any point, in any
combination, around the locked content tokens.

This directly contradicts the other half of the system: `factual_store.py:10-34`
(`RELATIONAL_KEYWORDS`) deliberately *boosts these same words* so they survive into the
captured spans (lines 194-199). So we spend effort capturing the source's relational
structure into the K/V span — and then let the model substitute its own connectives at
generation time, never enforcing the captured ones.

**Why it produces the failures.** Content is constrained; the *glue that assigns meaning*
is unconstrained. "EP2 \[is] codim 2 \[while] EP3 \[is] codim 3" and
"EP2 \[is] codim 3 \[while] EP3 \[is] codim 2" use the identical free helper scaffold.
This is the engine of "correct facts combined into incorrect explanations" and of
relationship inversion.

**Solution.** When SFA is active, **demote relational connectives out of the free-helper
set** and gate them: a contrastive/causal/copular token should only be allowed if it
actually appears adjacent to the currently-locked span in the source (quote-grounded
connective). Keep purely grammatical helpers ("the", "a", "of") free. This makes the
*relationship* as quote-grounded as the *content*.

---

### RC3 — Descriptors are random projections of max-pooled layer-0 keys (surface bags)

**Mechanism.** A span's descriptor is
`span_K[0].max(dim=1).values.mean(dim=0) @ W_proj.T` (`factual_store.py:290-292`), where
`W_proj` is a random matrix and `span_K[0]` is the **layer-0** key — essentially
pre-contextualization token/position embedding. Max-pool over positions then mean over
heads collapses the span to a token-presence signature.

**Why it produces the failures.** Two spans built from the same rare tokens in different
relational arrangements produce **near-identical descriptors**. "EP2 has codimension 2"
and "EP3 has codimension 3" share {codimension, has} and the rare entity tokens differ
only by one token that max-pool may not preserve. Query similarity therefore cannot
discriminate *which entity owns the value*. The retriever returns both with ~equal score;
the LM picks arbitrarily or averages. This is the mechanism behind "correct codimension
values + incorrect explanation of codimension" and "distinctions blur."

**Solution.** Make descriptors **discriminative on the entity axis**: (a) use a
contextualized (post-attention, mid/late layer) key rather than layer-0; and/or (b)
concatenate the span's resolved `entity_id`/prime token into the descriptor so two spans
that differ only by owning-entity separate in descriptor space. Random `W_proj` is fine
for hashing, but the *input* to it must already encode the distinction we need to
preserve.

---

### RC4 — Entity assignment by token-overlap collapses related concepts

**Mechanism.** Each non-prime span is assigned to the prime with the highest token
overlap (`factual_store.py:332-364`, Jaccard-style `|entry ∩ prime| / |entry|`).
Cross-entity edges are then dampened 0.3× (`factual_store.py:371`, `CROSS_ENTITY_DAMPEN`).

**Why it produces the failures.** For genuinely related entities — EP2 vs EP3, both
"exceptional points," both described with {eigenvector, codimension, branch point,
Riemann sheet} — the shared vocabulary *dominates* the overlap score. An EP3 property
span can score higher overlap with the EP2 prime than with its own EP3 prime, because the
shared structural tokens outnumber the one distinguishing token. The property is then
**mis-bound at the graph level, before any VSL filter runs.** Downstream entity-gating
cannot recover from a wrong `entity_id`; it faithfully enforces the wrong binding.

**Secondary regression (introduced last round).** The entity-tagging block added to
`diffkv_attention.py` recomputes each sequence's entity by **positional proximity**
(`nearest = min(prime_positions, key=|p - seq_start|)`), *ignoring* the
`entry.entity_id` already computed by token-overlap in the store. Positional proximity is
strictly worse for interleaved comparison text ("EP2 is X while EP3 is Y", where EP3's
property is positionally closer to the EP2 prime) — the exact case the store's
token-overlap logic was written to fix. The C++ mirror in `main.cpp` has the same
positional recompute. **These should read `fe.entity_id` directly.**

**Solution.** (1) Use the stored `entry.entity_id` everywhere instead of recomputing by
position. (2) Strengthen entity assignment so distinguishing tokens dominate shared ones:
weight overlap by IDF (a shared low-IDF "codimension" should count far less than the rare
entity-name token), and require the *prime's own distinguishing token* to be present
before assigning. (3) Consider hard cross-entity edge removal (not 0.3× dampening) for the
property layer, keeping a separate explicit "comparison" edge type for cross-entity links
needed by comparison queries.

---

### RC5 — Contrastive Category Anchor never fires on balanced comparisons

**Mechanism.** The anchor only commits to an entity when a single prime dominates: either
exactly one prime is active, or one prime has ≥2× the recent-token overlap of the other
(`diffkv_attention.py:794-800`; C++ `main.cpp` mirror). A balanced "compare A and B"
question keeps both primes roughly equal, so the 2× condition is never met,
`effective_prime_pos` stays unset, and `current_entity_id` stays `-1`.

**Why it produces the failures.** With `current_entity_id == -1`, the entity-filtered VSL
fallback degrades to "all sequence starts allowed" (by design — no context yet). So in
exactly the questions where binding matters most (comparisons), **no entity constraint is
ever applied** and both entities' spans interleave freely. This matches the report:
comparison/“multiple degeneracy types” questions are the worst-affected.

**Solution.** Replace "pick a dominant entity" with **explicit comparison mode**: when ≥2
primes are active and balanced, do not try to choose one — instead structure generation
as deterministic per-entity blocks (all of entity A's triples, then all of entity B's),
or round-robin with a hard entity lock per block. The anchor's job in a comparison should
be *segmentation*, not *winner selection*.

---

### RC6 — Top-5 / 5% truncation drops properties in multi-entity answers

**Mechanism.** Salience selection keeps only the top ~5% of tokens
(`factual_store.py:204-213`), and `query()` returns only the top 5 entries
(`factual_store.py:504-507`). The decode-time coherence cap further trims to 8 sequences
(`diffkv_attention.py:816-826`).

**Why it produces the failures.** A comparison needing 3 properties × 2 entities = 6
property spans plus 2 prime spans already exceeds the top-5 return. Whichever entity's
properties fall below the cut are simply **absent** from the factual set, so the model
fills the gap with its parametric prior — i.e., "generic explanatory text when precise
relationships are missing," verbatim from the report.

**Solution.** Make the cap **scale with the number of active entities** (e.g.,
`K = base + per_entity * num_active_primes`) so a 2-entity comparison gets a proportionally
larger budget, and guarantee a minimum quota of property spans *per active entity* rather
than a global top-K that one entity can monopolize.

---

### RC7 — "Prime" = any span with a rare token, not a curated entity

**Mechanism.** A span is flagged prime if it sits in a `semantic_prime_slot` *or* simply
contains a token with IDF ≥ 3.0 (`factual_store.py:319-327`).

**Why it produces the failures.** Rare-token = prime means a *property* span that happens
to contain a rare term ("Berry phase", "Puiseux series") becomes its **own entity**. An
entity's properties then scatter across several phantom primes, fragmenting the subgraph
the entity-filter relies on. Entity assignment (RC4) then has too many, too-granular
primes to bind against.

**Solution.** Separate "is rare/salient" from "is an entity." Promote to prime only spans
that are *referred back to* (high Eagle lookback `R`, already computed at
`factual_store.py:112-140`) — entities are things the rest of the document points at, not
merely things that are rare. This yields fewer, cleaner entities and a stable subgraph.

---

### RC8 — VSL is a sequence reproducer, not a relation validator

**Mechanism.** VSL enforces token *order within one captured span* and gates which spans
may *start* (`factual_alignment.py:get_allowed_tokens_vsl` / `update_vsl_state`). It has
no cross-span consistency check.

**Why it produces the failures.** Even with perfect spans, VSL cannot detect that the
model just emitted entity A's name and then entered entity B's property span — *as long as
B's start token was allowed*. It guarantees verbatim sub-strings, not correct *bindings
between* sub-strings.

**Solution.** Add a lightweight **binding validator** on top of VSL: when about to emit a
content token that completes an `(entity, property)` pairing, verify that pairing exists
as a stored triple (from RC1); if not, apply a hard penalty. This is the generation-time
analogue of "quote-grounded verification."

---

## 3. Why the previous fixes plateaued

All of: sequence-start-only fallback, helper-threshold 4→12, graduated soft/hard VSL,
entity-subgraph filtering, coherence cap — operate on **which tokens may be emitted next**.
They are necessary hygiene, and they did move us from "retrieval failure" to "binding
failure." But they share a ceiling: **they gate a token stream against a set of spans that
contain no relational structure.** You cannot gate your way to a correct binding that was
never stored. The next gains require representing the relation itself (RC1) and grounding
the connectives that express it (RC2).

---

## 4. Recommended Program (priority order)

**P0 — Structural (unblocks everything else)**
1. **Typed directed triples** (RC1): build `(subject_entity, relation_span, value_span)`
   at prefill; retrieve triples; lock VSL to triple ordering.
2. **Quote-grounded connectives** (RC2): demote contrastive/causal/copular words from the
   free-helper set under SFA; allow only source-adjacent connectives.

**P1 — Correctness of the existing entity layer (cheap, high impact)**
3. Use stored `entry.entity_id` instead of positional recompute in `diffkv_attention.py`
   and `main.cpp` (RC4 regression — this is a near-free fix).
4. IDF-weighted entity assignment + require the prime's distinguishing token (RC4).
5. Explicit **comparison mode** with per-entity blocks instead of 2× winner selection
   (RC5).
6. Entity-proportional retrieval budget with per-entity property quota (RC6).

**P2 — Representation quality**
7. Discriminative descriptors: contextualized key + entity token in the descriptor input
   (RC3).
8. Reference-based prime promotion via Eagle lookback, decoupled from rarity (RC7).
9. Generation-time binding validator over the triple store (RC8).

---

## 5. Evaluation harness (matches the report's requested direction)

Stop scoring on ROUGE/keyword recall; score on **bindings**. Concretely, hand-label a
small set of source triples and measure:

- **Entity-property matching accuracy** — for each generated `(entity, property)` pair, is
  it a true source triple? (precision) and did we cover the gold pairs? (recall)
- **Relation-extraction F1** on generated text vs gold triples.
- **Contradiction detection rate** — fraction of generations containing a pair that
  contradicts a gold triple (this is the metric that captures "inversion").
- **Comparison-table cell accuracy** — render the answer as an entity×property grid and
  score per cell; directly targets the comparison failures.
- **Concept→attribute mapping accuracy** — same as entity-property but for non-entity
  attributes (codimension value, eigenvector behavior, etc.).
- **Quote-grounded verification rate** — fraction of asserted bindings that are
  substring-verifiable against the source.

A regression suite of ~20 comparison questions scored on comparison-table cell accuracy
would give the tightest signal on the exact failure mode described.

---

## 6. Single-sentence summary

We built an excellent retriever of *what* the document says and never built a representation
of *how those pieces relate*; the model recalls the pieces and is then forced to re-invent
the relationships through unconstrained connective words — so the fix is to make relations
first-class (stored, typed, directed) and to ground the connectives that express them,
rather than to keep tightening token gates over relation-free spans.
