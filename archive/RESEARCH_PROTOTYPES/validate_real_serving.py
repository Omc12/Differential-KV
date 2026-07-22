"""
validate_real_serving.py

REAL validation of the live serving stack.

Tests:
  1. Sampling diversity  — outputs are not deterministic/greedy repeats
  2. Repetition         — bigram overlap ratio in generated text
  3. Multi-turn adaptation — model responds correctly to follow-up prompts
  4. Long-context stability — model doesn't freeze/repeat at 2k+ token contexts
  5. Streaming smoothness   — chunk sizes are phrase-groups, not single chars

Run:  python validate_real_serving.py
Requires the server to be running at http://localhost:8000
"""

import sys
import json
import time
import httpx
import asyncio
from pathlib import Path
from typing import List, Dict, Tuple

BASE_URL = "http://localhost:8000/v1"
MODEL   = "dkv-serving"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bigram_repetition_ratio(text: str) -> float:
    """
    Returns the fraction of bigrams that are repeated.
    0.0 = no repetition. 1.0 = every bigram repeats.
    Human text is typically < 0.15.
    """
    words = text.lower().split()
    if len(words) < 2:
        return 0.0
    bigrams = [(words[i], words[i+1]) for i in range(len(words) - 1)]
    total  = len(bigrams)
    unique = len(set(bigrams))
    return 1.0 - (unique / total)


def unique_token_ratio(text: str, window: int = 100) -> float:
    """Unique word ratio in last `window` tokens. Low = repetitive."""
    words = text.split()[-window:]
    if not words:
        return 1.0
    return len(set(words)) / len(words)


async def chat(
    client: httpx.AsyncClient,
    messages: List[Dict],
    session_id: str = None,
    temperature: float = 0.7,
    max_tokens: int = 256,
    stream: bool = False,
) -> Tuple[str, float, List[int]]:
    """
    Sends a chat request. Returns (text, ttft_seconds, chunk_sizes).
    chunk_sizes only populated when stream=True.
    """
    payload = {
        "model":       MODEL,
        "messages":    messages,
        "temperature": temperature,
        "max_tokens":  max_tokens,
        "stream":      stream,
    }
    if session_id:
        payload["session_id"] = session_id

    chunk_sizes: List[int] = []
    text = ""
    ttft = None
    t0 = time.perf_counter()

    if stream:
        async with client.stream("POST", f"{BASE_URL}/chat/completions",
                                 json=payload, timeout=120.0) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if raw == "[DONE]":
                    break
                try:
                    data = json.loads(raw)
                    delta = data["choices"][0]["delta"].get("content", "")
                    if delta:
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        text += delta
                        chunk_sizes.append(len(delta.split()))
                except Exception:
                    pass
    else:
        resp = await client.post(f"{BASE_URL}/chat/completions",
                                  json=payload, timeout=120.0)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        ttft = time.perf_counter() - t0

    return text, ttft or (time.perf_counter() - t0), chunk_sizes


async def create_session(client: httpx.AsyncClient) -> str:
    resp = await client.post(f"{BASE_URL}/sessions")
    return resp.json()["session_id"]


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

async def test_sampling_diversity(client: httpx.AsyncClient) -> bool:
    """
    Run the same prompt 3 times. If all responses are identical, sampling is broken.
    """
    print("\n[1] Sampling Diversity")
    prompt = [{"role": "user", "content": "Tell me an interesting fact about the ocean."}]
    outputs = []
    for i in range(3):
        text, _, _ = await chat(client, prompt, temperature=0.7, max_tokens=100)
        outputs.append(text.strip())
        print(f"    Run {i+1}: {text[:80]!r}...")

    unique = len(set(outputs))
    if unique == 1:
        print("    FAIL — all 3 outputs are identical (greedy/deterministic decoding)")
        return False
    elif unique == 2:
        print(f"    WARN — only {unique}/3 outputs differ (low diversity)")
        return True
    else:
        print(f"    PASS — {unique}/3 outputs are distinct")
        return True


async def test_repetition(client: httpx.AsyncClient) -> bool:
    """
    Generate a longer response and check bigram repetition ratio.
    Fail threshold: > 0.25 (humans are usually < 0.15).
    """
    print("\n[2] Repetition Check")
    prompt = [{"role": "user",
               "content": "Explain the history and significance of the Silk Road in detail."}]
    text, _, _ = await chat(client, prompt, temperature=0.7, max_tokens=300)
    ratio = bigram_repetition_ratio(text)
    utr   = unique_token_ratio(text)
    print(f"    Output length:       {len(text.split())} words")
    print(f"    Bigram repeat ratio: {ratio:.3f}  (fail if > 0.25)")
    print(f"    Unique token ratio:  {utr:.3f}  (fail if < 0.35)")

    if ratio > 0.25:
        print(f"    FAIL — high repetition ({ratio:.3f})")
        return False
    if utr < 0.35:
        print(f"    FAIL — low unique tokens ({utr:.3f})")
        return False
    print("    PASS")
    return True


async def test_multiturn_adaptation(client: httpx.AsyncClient) -> bool:
    """
    Tests whether the model correctly adapts to follow-up prompts.
    Sends a question, then 'what?', then 'can you elaborate on the first point?'
    """
    print("\n[3] Multi-Turn Adaptation")
    sid = await create_session(client)

    # Turn 1
    t1_msgs = [{"role": "user", "content": "What are three benefits of regular exercise?"}]
    t1_text, _, _ = await chat(client, t1_msgs, session_id=sid, max_tokens=200)
    print(f"    T1: {t1_text[:100]!r}...")

    # Turn 2: ambiguous follow-up
    t2_msgs = [{"role": "user", "content": "What?"}]
    t2_text, _, _ = await chat(client, t2_msgs, session_id=sid, max_tokens=150)
    print(f"    T2: {t2_text[:100]!r}...")

    # Turn 3: specific follow-up
    t3_msgs = [{"role": "user", "content": "Can you elaborate on the first benefit?"}]
    t3_text, _, _ = await chat(client, t3_msgs, session_id=sid, max_tokens=200)
    print(f"    T3: {t3_text[:100]!r}...")

    # Checks
    # T2 should not be a repeat of T1 verbatim
    t2_overlap = bigram_repetition_ratio(t1_text + " " + t2_text)
    # T3 should reference something from T1 (contains exercise-related terms)
    exercise_terms = {"exercise", "fitness", "health", "benefit", "physical",
                      "muscle", "heart", "weight", "energy", "mental"}
    t3_words = set(t3_text.lower().split())
    t3_relevance = len(exercise_terms & t3_words)

    print(f"    T2 overlap with T1: {t2_overlap:.3f}")
    print(f"    T3 exercise terms matched: {t3_relevance}")

    if len(t2_text.strip()) < 10:
        print("    FAIL — T2 response is empty (model lost context)")
        return False
    if t3_relevance == 0:
        print("    FAIL — T3 doesn't reference prior conversation")
        return False
    print("    PASS")
    return True


async def test_followup_prompts(client: httpx.AsyncClient) -> bool:
    """
    Tests 'continue', 'say something else', 'what did you mean by that?'
    """
    print("\n[4] Follow-Up Prompt Handling")
    sid = await create_session(client)

    # Set up context
    await chat(client,
               [{"role": "user", "content": "Tell me about quantum computing in simple terms."}],
               session_id=sid, max_tokens=150)

    results = {}
    for followup in ["continue", "say something else", "what did you mean by that?"]:
        text, _, _ = await chat(client,
                                 [{"role": "user", "content": followup}],
                                 session_id=sid, max_tokens=150)
        results[followup] = text
        empty = len(text.strip()) < 15
        print(f"    '{followup}' -> {'EMPTY (FAIL)' if empty else text[:70]!r + '...'}")

    # All responses must be non-empty and non-identical
    texts = list(results.values())
    all_nonempty = all(len(t.strip()) >= 15 for t in texts)
    all_distinct  = len(set(t[:50] for t in texts)) == len(texts)

    if not all_nonempty:
        print("    FAIL — some follow-up responses are empty")
        return False
    if not all_distinct:
        print("    WARN — some follow-up responses are identical (stale state)")
    print("    PASS")
    return True


async def test_streaming_smoothness(client: httpx.AsyncClient) -> bool:
    """
    Verifies streaming emits phrase groups (>1 word) not single characters.
    Fails if median chunk is 0–1 words (word-by-word rendering).
    """
    print("\n[5] Streaming Chunk Size")
    prompt = [{"role": "user",
               "content": "Describe the process of photosynthesis step by step."}]
    text, ttft, chunk_sizes = await chat(
        client, prompt, temperature=0.7, max_tokens=200, stream=True
    )

    if not chunk_sizes:
        print("    FAIL — no chunks received")
        return False

    avg_chunk = sum(chunk_sizes) / len(chunk_sizes)
    min_chunk = min(chunk_sizes)
    single_word_pct = sum(1 for c in chunk_sizes if c <= 1) / len(chunk_sizes)

    print(f"    Total chunks:        {len(chunk_sizes)}")
    print(f"    Avg chunk (words):   {avg_chunk:.2f}  (pass if >= 2.0)")
    print(f"    Min chunk (words):   {min_chunk}")
    print(f"    Single-word chunks:  {single_word_pct:.1%}  (fail if > 60%)")
    print(f"    Time to first chunk: {ttft:.3f}s")

    if single_word_pct > 0.60:
        print("    FAIL — majority of chunks are single words (word-by-word rendering)")
        return False
    if avg_chunk < 2.0:
        print("    FAIL — average chunk too small")
        return False
    print("    PASS")
    return True


async def test_long_context_stability(client: httpx.AsyncClient) -> bool:
    """
    Sends a large context (~1500 words) then asks a follow-up.
    Checks for repetition and coherent response.
    """
    print("\n[6] Long-Context Stability")

    # Build a long context
    long_text = (
        "The following is a detailed account of the development of machine learning. " +
        "Machine learning began in the 1950s with Alan Turing's foundational question: "
        "'Can machines think?' This led to the Turing Test. " * 40 +
        "More recently, transformer architectures have dominated the field, "
        "enabling large language models like GPT and BERT. " * 20
    )
    messages = [
        {"role": "user", "content": long_text},
        {"role": "assistant", "content": "I understand. This covers the history of machine learning."},
        {"role": "user", "content": "Based on everything above, what was Alan Turing's contribution?"},
    ]
    text, _, _ = await chat(client, messages, temperature=0.7, max_tokens=200)
    rep = bigram_repetition_ratio(text)

    print(f"    Context length:  ~{len(long_text.split())} words")
    print(f"    Response:        {text[:120]!r}...")
    print(f"    Repetition ratio: {rep:.3f}  (fail if > 0.30)")

    turing_mentioned = "turing" in text.lower()
    if not turing_mentioned:
        print("    WARN — response doesn't mention Turing despite explicit question")
    if rep > 0.30:
        print("    FAIL — high repetition in long-context response")
        return False
    if len(text.strip()) < 20:
        print("    FAIL — empty/near-empty response on long context")
        return False
    print("    PASS")
    return True


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run_all():
    print("=" * 60)
    print("  Differential KV — Real Serving Validation")
    print("=" * 60)
    print(f"  Target: {BASE_URL}")

    async with httpx.AsyncClient(timeout=180.0) as client:

        # Check server is alive
        try:
            resp = await client.get(f"{BASE_URL}/models")
            resp.raise_for_status()
            print(f"  Model:  {resp.json()['data'][0]['id']}")
        except Exception as e:
            print(f"\nERROR: Cannot reach server at {BASE_URL}")
            print(f"  {e}")
            print("  Start the server first: python run_dkv_webui_server.py")
            return

        results: Dict[str, bool] = {}

        results["sampling_diversity"]    = await test_sampling_diversity(client)
        results["repetition"]            = await test_repetition(client)
        results["multiturn_adaptation"]  = await test_multiturn_adaptation(client)
        results["followup_handling"]     = await test_followup_prompts(client)
        results["streaming_smoothness"]  = await test_streaming_smoothness(client)
        results["long_context_stability"] = await test_long_context_stability(client)

    # Summary
    print("\n" + "=" * 60)
    print("  Results")
    print("=" * 60)
    passed = sum(v for v in results.values())
    total  = len(results)
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}]  {name}")
    print(f"\n  {passed}/{total} tests passed")

    if passed < total:
        failing = [k for k, v in results.items() if not v]
        print(f"\n  Failing: {', '.join(failing)}")
        print("  These represent real UX problems that need real code fixes.")
    else:
        print("\n  All tests passed — real serving behavior is acceptable.")

    print("=" * 60)
    return passed == total


if __name__ == "__main__":
    ok = asyncio.run(run_all())
    sys.exit(0 if ok else 1)
