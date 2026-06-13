# native_core/srl/factual_alignment.py
import torch
from typing import Optional, Set

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


def get_allowed_tokens_vsl(srl_state, helper_ids: set) -> set:
    allowed = set(helper_ids)
    
    factual_sequences = getattr(srl_state, "current_step_factual_sequences", None) or []
    active_candidates = getattr(srl_state, "vsl_active_candidates", None) or []
    
    has_active_lock = False
    for suffix in active_candidates:
        if suffix:
            allowed.add(suffix[0])
            has_active_lock = True
            
    if not has_active_lock:
        for seq in factual_sequences:
            if seq:
                allowed.update(seq)
                
    return allowed


def update_vsl_state(token_id: int, srl_state, helper_ids: set):
    if token_id in helper_ids:
        srl_state.vsl_consecutive_helpers = getattr(srl_state, "vsl_consecutive_helpers", 0) + 1
        # Threshold lowered 6 → 4: with tight 5%-selection factual sequences, the model
        # should stay locked to a source phrase. 4 consecutive helpers signals real drift.
        if srl_state.vsl_consecutive_helpers >= 4:
            srl_state.vsl_active_candidates = []
        return
        
    factual_sequences = getattr(srl_state, "current_step_factual_sequences", None) or []
    active_candidates = getattr(srl_state, "vsl_active_candidates", None) or []
    
    new_candidates = []
    
    for suffix in active_candidates:
        if suffix and suffix[0] == token_id:
            new_candidates.append(suffix[1:])
            
    has_active_lock = any(len(suffix) > 0 for suffix in active_candidates)
    if not new_candidates and not has_active_lock:
        for seq in factual_sequences:
            for j, t_id in enumerate(seq):
                if t_id == token_id:
                    new_candidates.append(seq[j+1:])
                    
    srl_state.vsl_active_candidates = new_candidates
    srl_state.vsl_consecutive_helpers = 0


def is_token_id_allowed(tok_id: int, srl_state, last_token: Optional[int], tokenizer) -> bool:
    helper_ids = get_helper_token_ids(tokenizer)
    allowed = get_allowed_tokens_vsl(srl_state, helper_ids)
    return tok_id in allowed
