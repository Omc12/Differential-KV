# native_core/srl/factual_alignment.py
import torch
from typing import Optional, Set

# Words that CARRY relational meaning: copular verbs, contrastive conjunctions,
# causal connectives.  Under SFA + active lock these are excluded from free
# helpers so the model cannot fabricate its own binding phrase; it must instead
# enter a triple sequence (bridge + value) that was grounded in the source text.
# In unlocked fallback mode they remain available so the model can start triple
# sequences whose first token is a relational word (e.g. "is", "has", "whereas").
RELATIONAL_BINDING_WORDS = {
    # Copular / attribution verbs
    "is", "are", "was", "were", "has", "have", "had",
    "exhibits", "possesses", "contains", "involves", "requires", "lacks", "features",
    # Contrastive conjunctions
    "whereas", "while", "although", "but", "however", "yet", "though",
    "notwithstanding", "nevertheless", "nonetheless", "conversely",
    "unlike", "contrast", "instead", "rather",
    # Causal connectives
    "because", "since", "therefore", "hence", "thus", "consequently",
    "accordingly",
    # Comparative
    "than",
}

# Curated helper vocabulary for grammatical and structural tokens
ALLOWED_HELPER_WORDS = {
    # Pronouns & Determiners
    "i", "me", "my", "myself", "we", "us", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "who", "whom", "whose", "which", "that", "this", "these", "those", "each", "every", "some", "any", "no", "all", "both",
    "either", "neither", "another", "other", "such", "what", "a", "an", "the",

    # Prepositions & Conjunctions
    "of", "in", "to", "for", "with", "on", "at", "by", "from", "about", "as", "into", "through", "during", "before", "after",
    "above", "below", "up", "down", "over", "under", "between", "among", "out", "off", "within", "without", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how", "and", "or", "but", "so", "yet", "nor", "although",
    "because", "since", "unless", "until", "while", "whereas", "if", "else", "than",

    # Verbs
    "is", "was", "were", "are", "be", "been", "being", "am", "have", "has", "had", "having", "do", "does", "did", "doing",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must", "say", "says", "said", "saying",
    "state", "states", "stated", "stating", "give", "gives", "given", "giving", "show", "shows", "shown", "showing",
    "write", "writes", "written", "writing", "read", "reads", "mention", "mentions", "mentioned", "describe", "describes",
    "described", "refer", "refers", "referred", "contain", "contains", "contained", "include", "includes", "included",
    "follow", "follows", "following", "followed", "find", "finds", "found", "express", "expresses", "expressed",

    # Nouns & Adjectives (meta/structural)
    "source", "document", "text", "passage", "file", "formula", "relation", "equation", "equations",
    "theorem", "definition", "fact", "facts", "retrieval", "information", "detail", "details", "exact", "exactly",
    "correct", "correctly", "faithful", "faithfully", "verbatim", "missing", "present", "clear", "clearly",
    "uncertain", "certain", "provide", "provided", "provides", "note", "noted", "notes",

    # Common helper adverbs/adjectives
    "not", "only", "very", "too", "just", "well", "also", "now", "first", "second", "third", "one", "two", "three",

    # Mathematical and relational verbs/nouns
    "case", "cases", "correspond", "corresponds", "corresponding", "example", "examples", "result", "results",
    "value", "values", "number", "numbers", "term", "terms", "word", "words", "mean", "means", "meant", "meaning",
    "define", "defines", "defined", "definition", "definitions", "represent", "represents", "represented",
    "explanation", "explanations", "statement", "statements", "use", "uses", "used", "using", "make", "makes",
    "made", "making", "take", "takes", "taken", "taking", "part", "parts", "point", "points", "set", "sets"
}

def advance_comparison_entity(comparison_entities, active_idx, covered, recent_tokens,
                              prime_tokens_by_entity, prop_tokens_by_entity):
    """RC5 — sequence a comparison as deterministic per-entity blocks.

    Rather than letting two balanced entities interleave (the worst-affected
    case in the report), we lock generation to ONE entity at a time and advance
    only once that entity has been substantively covered — i.e. both its prime
    AND at least one of its property tokens have appeared in recent output.

    Returns (new_active_idx, new_covered).  Pure; unit-tested.
    """
    if not comparison_entities:
        return active_idx, covered
    active_idx = max(0, min(active_idx, len(comparison_entities) - 1))
    covered = set(covered)
    recent = set(recent_tokens)
    active_eid = comparison_entities[active_idx]

    prime_toks = prime_tokens_by_entity.get(active_eid, set())
    prop_toks = prop_tokens_by_entity.get(active_eid, set())
    prime_seen = bool(prime_toks & recent)
    prop_seen = bool(prop_toks & recent)

    if prime_seen and prop_seen:
        covered.add(active_eid)
        # Advance to the next entity not yet covered (clamp at the last block).
        for nxt in range(active_idx + 1, len(comparison_entities)):
            if comparison_entities[nxt] not in covered:
                return nxt, covered
        # All remaining covered — stay on the last block (don't relax to interleave).
    return active_idx, covered


def compute_entity_token_license(sequences, entity_ids, is_prime_flags, current_entity):
    """RC8 — generation-time binding validator support.

    Given the per-step factual sequences (which already include RC1 triple
    sequences, each tagged with its owning entity), split tokens into:
      - licensed: tokens the current entity is allowed to emit (its own spans /
        triples, plus entity-agnostic and prime spans), and
      - foreign:  tokens that belong exclusively to OTHER entities.

    Emitting a foreign token while locked to current_entity is exactly the
    "EP2 has codimension 3" inversion; the caller penalises/masks the foreign
    set.  Returns (licensed, foreign).  Pure; unit-tested.
    """
    licensed = set()
    other = set()
    for i, seq in enumerate(sequences):
        if not seq:
            continue
        eid = entity_ids[i] if i < len(entity_ids) else -1
        isp = is_prime_flags[i] if i < len(is_prime_flags) else False
        if isp or eid == -1 or eid == current_entity:
            licensed.update(seq)
        else:
            other.update(seq)
    foreign = other - licensed
    return licensed, foreign


def get_structural_helper_token_ids(tokenizer) -> set:
    """Full helpers minus relational binders — used under SFA + active lock (RC2)."""
    if hasattr(tokenizer, "_structural_helper_token_ids_cache"):
        return tokenizer._structural_helper_token_ids_cache

    full = get_helper_token_ids(tokenizer)
    # Build the set of relational token IDs to exclude
    relational_ids: set = set()
    for word in RELATIONAL_BINDING_WORDS:
        try:
            for tok_id in tokenizer.encode(word, add_special_tokens=False):
                relational_ids.add(tok_id)
            for tok_id in tokenizer.encode(" " + word, add_special_tokens=False):
                relational_ids.add(tok_id)
        except Exception:
            pass
    # Fallback: scan vocab
    if hasattr(tokenizer, "get_vocab"):
        try:
            for token_str, tok_id in tokenizer.get_vocab().items():
                cleaned = token_str.replace("Ġ", "").replace(" ", "").replace("</w>", "")
                cleaned = "".join(c for c in cleaned if c.isalnum()).strip().lower()
                if cleaned in RELATIONAL_BINDING_WORDS:
                    relational_ids.add(tok_id)
        except Exception:
            pass

    structural = full - relational_ids
    tokenizer._structural_helper_token_ids_cache = structural
    return structural


def get_helper_token_ids(tokenizer) -> set:
    if hasattr(tokenizer, "_helper_token_ids_cache"):
        return tokenizer._helper_token_ids_cache

    helper_ids = set()

    if hasattr(tokenizer, "get_vocab"):
        try:
            vocab = tokenizer.get_vocab()
            for token_str, tok_id in vocab.items():
                cleaned = token_str.replace("Ġ", "").replace(" ", "").replace("</w>", "")
                cleaned = "".join(c for c in cleaned if c.isalnum()).strip().lower()
                if not cleaned or cleaned in ALLOWED_HELPER_WORDS:
                    helper_ids.add(tok_id)
        except Exception:
            pass

    if not helper_ids:
        if hasattr(tokenizer, "encode"):
            for word in ALLOWED_HELPER_WORDS:
                try:
                    t_ids = tokenizer.encode(word, add_special_tokens=False)
                    helper_ids.update(t_ids)
                    t_ids_space = tokenizer.encode(" " + word, add_special_tokens=False)
                    helper_ids.update(t_ids_space)
                except Exception:
                    pass

        try:
            vocab_size = tokenizer.vocab_size
        except AttributeError:
            try:
                vocab_size = len(tokenizer)
            except Exception:
                vocab_size = 300

        for tok_id in range(vocab_size):
            try:
                text = tokenizer.decode([tok_id])
                cleaned = "".join(c for c in text if c.isalnum()).strip().lower()
                if not cleaned or cleaned in ALLOWED_HELPER_WORDS:
                    helper_ids.add(tok_id)
            except Exception:
                pass

    tokenizer._helper_token_ids_cache = helper_ids
    return helper_ids


def get_allowed_tokens_vsl(srl_state, helper_ids: set,
                           structural_helper_ids: Optional[Set[int]] = None,
                           sfa_active: bool = False) -> set:
    """
    Return the set of allowed token IDs for the current decode step.

    LOCK ACTIVE:
      RC2 — when SFA is active, use structural_helper_ids (full helpers minus
      relational binders) so "is", "has", "whereas", "because" etc. cannot be
      freely emitted between locked content tokens.  They must come from a triple
      sequence (bridge + value) grounded in the source text.
      allowed = base_helpers ∪ {suffix[0] for each active suffix}

    NO ACTIVE LOCK (entity-filtered):
      Full helpers always available — triple sequences start with relation words
      ("is"/"has") so the model needs them in unlocked fallback to enter triples.
      allowed = helper_ids ∪ {seq[0] for same-entity / prime sequences}
    """
    factual_sequences = getattr(srl_state, "current_step_factual_sequences", None) or []
    active_candidates = getattr(srl_state, "vsl_active_candidates", None) or []
    entity_ids    = getattr(srl_state, "current_step_sequence_entity_ids", [])
    is_prime_list = getattr(srl_state, "current_step_sequence_is_prime", [])
    # RC2: per-sequence source prefix tokens (the 3 tokens preceding each span in
    # the source document).  Used to quote-ground relational connectives: a
    # contrastive/causal/copular word may only bridge into a span if the source
    # actually placed it there.  Parallel to factual_sequences; may be shorter.
    seq_prefixes  = getattr(srl_state, "current_step_sequence_prefixes", [])
    current_entity = getattr(srl_state, "current_entity_id", -1)
    dual_mode = getattr(srl_state, "dual_entity_mode", False)
    dual_ids  = getattr(srl_state, "dual_entity_ids", [])

    has_active_lock = any(bool(s) for s in active_candidates)

    # RC2: restrict to structural helpers when SFA + lock active
    if sfa_active and has_active_lock and structural_helper_ids is not None:
        base_helpers = structural_helper_ids
    else:
        base_helpers = helper_ids

    allowed = set(base_helpers)
    for suffix in active_candidates:
        if suffix:
            allowed.add(suffix[0])

    if not has_active_lock:
        # Unlocked fallback — restore full helpers, add entity-filtered starts.
        allowed = set(helper_ids)

        # Single pass over candidate sequences: collect the start tokens we may
        # enter (entity-filtered) and the source-adjacent connectives that bridge
        # into them (their prefix tokens).
        enterable_starts: set = set()
        grounded_connectives: set = set()
        for i, seq in enumerate(factual_sequences):
            if not seq:
                continue
            seq_entity   = entity_ids[i]  if i < len(entity_ids)    else -1
            seq_is_prime = is_prime_list[i] if i < len(is_prime_list) else False

            if seq_is_prime or seq_entity == -1:
                enterable = True
            elif current_entity != -1:
                enterable = (seq_entity == current_entity)
            elif dual_mode and dual_ids:
                enterable = (seq_entity in dual_ids)
            else:
                enterable = True

            if enterable:
                enterable_starts.add(seq[0])
                # The tokens that preceded this span in the source are the only
                # connectives allowed to bridge into it (RC2 quote-grounding).
                if i < len(seq_prefixes) and seq_prefixes[i]:
                    grounded_connectives.update(seq_prefixes[i])

        # RC2 — Quote-grounded connective gate.  Relational binding words
        # ("is", "has", "because", "whereas", "while", …) are demoted from the
        # free-helper set under SFA and re-admitted ONLY where the source grounds
        # them: either they begin a captured sequence (a triple bridge) or they
        # were source-adjacent to a span we may now enter.  This stops the model
        # from inventing its own connective scaffold around correct content —
        # the engine of Relationship Hallucination (failure 3), Definition Drift
        # (failure 1) and relationship inversion in comparisons.
        if sfa_active and structural_helper_ids is not None:
            relational_ids = helper_ids - structural_helper_ids
            grounded = enterable_starts | grounded_connectives
            for rid in relational_ids:
                if rid not in grounded:
                    allowed.discard(rid)

        allowed |= enterable_starts

    return allowed


def update_vsl_state(token_id: int, srl_state, helper_ids: set):
    """
    Advance the VSL lock state after a token is generated.

    Helpers pass through without advancing the lock. Threshold raised 4→12 so
    normal bridge phrases don't discard a valid lock prematurely.

    When starting a new lock (no prior lock, token matches a sequence start),
    record that sequence's entity_id as the new current_entity_id so subsequent
    fallback steps restrict to the same entity's sequences.
    """
    if token_id in helper_ids:
        srl_state.vsl_consecutive_helpers = getattr(srl_state, "vsl_consecutive_helpers", 0) + 1
        if srl_state.vsl_consecutive_helpers >= 12:
            srl_state.vsl_active_candidates = []
        return

    factual_sequences = getattr(srl_state, "current_step_factual_sequences", None) or []
    active_candidates = getattr(srl_state, "vsl_active_candidates", None) or []
    entity_ids    = getattr(srl_state, "current_step_sequence_entity_ids", [])

    new_candidates = []

    for suffix in active_candidates:
        if suffix and suffix[0] == token_id:
            new_candidates.append(suffix[1:])

    has_active_lock = any(len(suffix) > 0 for suffix in active_candidates)
    if not new_candidates and not has_active_lock:
        # In entity-filtered fallback mode only sequence-start tokens are reachable,
        # so we only look at seq[0] here.  Update entity context from the sequence
        # we are entering so the next fallback step is entity-consistent.
        for i, seq in enumerate(factual_sequences):
            if seq and seq[0] == token_id:
                new_candidates.append(seq[1:])
                seq_entity = entity_ids[i] if i < len(entity_ids) else -1
                if seq_entity != -1:
                    srl_state.current_entity_id = seq_entity

    srl_state.vsl_active_candidates = new_candidates
    srl_state.vsl_consecutive_helpers = 0


def is_token_id_allowed(tok_id: int, srl_state, last_token: Optional[int], tokenizer) -> bool:
    helper_ids = get_helper_token_ids(tokenizer)
    allowed = get_allowed_tokens_vsl(srl_state, helper_ids)
    return tok_id in allowed
