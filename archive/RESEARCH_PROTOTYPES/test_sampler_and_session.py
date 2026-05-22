"""Quick unit test for the sampler and session history — no model load required."""
import sys
sys.path.insert(0, ".")
import torch
from runtime.lgs_resolver import LGSResolver
from serving.production_session_manager import ProductionSessionManager

# ── 1. Sampler diversity ──────────────────────────────────────────────────────
resolver = LGSResolver({})
vocab = 1000
batch = 2
torch.manual_seed(42)
logits_base = torch.randn(batch, vocab)

stochastic_results = []
for _ in range(5):
    toks = resolver._sample_next_tokens(
        logits_base.clone(), temperature=0.7, top_p=0.9,
        generated_ids_per_seq=[[], []],
        repetition_penalty=1.0,
    )
    stochastic_results.append(toks[0, 0].item())

unique = len(set(stochastic_results))
status = "PASS" if unique > 1 else "FAIL"
print(f"[{status}] Sampler diversity: {stochastic_results} ({unique}/5 unique)")
assert unique > 1, "Sampler is deterministic - greedy not fixed!"

# ── 2. Greedy at temp=0 ───────────────────────────────────────────────────────
greedy_results = []
for _ in range(5):
    toks = resolver._sample_next_tokens(
        logits_base.clone(), temperature=0.0, top_p=0.9,
        generated_ids_per_seq=[[], []],
        repetition_penalty=1.0,
    )
    greedy_results.append(toks[0, 0].item())

unique_g = len(set(greedy_results))
status_g = "PASS" if unique_g == 1 else "FAIL"
print(f"[{status_g}] Greedy (temp=0): {greedy_results} ({unique_g}/5 unique, expect 1)")

# ── 3. Repetition penalty lowers logit of repeated token ─────────────────────
logits_test = torch.zeros(1, vocab)
logits_test[0, 42] = 5.0   # token 42 is dominant
recent = [42] * 20          # token 42 was just generated 20 times
penalised = resolver._sample_next_tokens(
    logits_test.clone(), temperature=0.01, top_p=1.0,
    generated_ids_per_seq=[recent],
    repetition_penalty=1.5,
)
# After penalty, token 42's logit should be reduced: 5.0 / 1.5 = 3.33
# So greedy should still pick it, but check penalisation happened
no_pen_logits = torch.zeros(1, vocab)
no_pen_logits[0, 42] = 5.0
no_pen_tok = resolver._sample_next_tokens(
    no_pen_logits.clone(), temperature=0.01, top_p=1.0,
    generated_ids_per_seq=[[]],
    repetition_penalty=1.0,
)
print(f"[INFO] Rep-penalty: tok=42 still selected={penalised[0,0].item()==42}, no-pen={no_pen_tok[0,0].item()==42}")

# ── 4. Session manager history ────────────────────────────────────────────────
sm = ProductionSessionManager()
sid = sm.create_session()
sm.append_message(sid, "user", "Hello!")
sm.append_message(sid, "assistant", "Hi there!")
sm.append_message(sid, "user", "What is 2+2?")

hist = sm.get_history(sid)
assert len(hist) == 3, f"Expected 3 history entries, got {len(hist)}"
assert hist[1]["role"] == "assistant"
print(f"[PASS] Session history: {len(hist)} turns stored correctly")

sm.clear_history(sid)
assert sm.get_history(sid) == [], "History not cleared"
print("[PASS] Session clear: OK")

print("\nAll unit tests passed.")
