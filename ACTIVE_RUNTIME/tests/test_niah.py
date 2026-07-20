import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def make_niah_prompt(tokenizer, context_length, depth, needle, question):
    filler = (
        "Quantum computing is a multidisciplinary field comprising aspects of computer science, "
        "physics, and mathematics that utilizes quantum mechanics to solve complex problems faster "
        "than on classical computers. The field of quantum computing includes hardware research and "
        "application development. Quantum computers are able to solve certain classes of problems "
        "much faster than classical computers by taking advantage of quantum mechanical effects, "
        "such as superposition and quantum entanglement. "
    )
    
    filler_tokens = tokenizer.encode(filler, add_special_tokens=False)
    needle_tokens = tokenizer.encode(needle + "\n", add_special_tokens=False)
    
    # Estimate remaining room for templates and questions
    target_filler_tokens = context_length - len(needle_tokens) - 100
    if target_filler_tokens < 0:
        target_filler_tokens = 100
        
    num_repeats = (target_filler_tokens // len(filler_tokens)) + 1
    all_filler_tokens = (filler_tokens * num_repeats)[:target_filler_tokens]
    
    insert_idx = int(len(all_filler_tokens) * depth)
    part1_tokens = all_filler_tokens[:insert_idx]
    part2_tokens = all_filler_tokens[insert_idx:]
    
    part1_text = tokenizer.decode(part1_tokens)
    part2_text = tokenizer.decode(part2_tokens)
    
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        + part1_text + "\n"
        + needle + "\n"
        + part2_text + "\n\n"
        + question + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return prompt

@pytest.mark.parametrize("depth", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("context_len", [4000, 8000])
def test_niah_depths(depth, context_len):
    from serving.hf_diffkv_wrapper import DiffKVHFWrapper
    
    MODEL = os.environ.get("DIFFKV_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    
    # Validate the SHIPPING decode path. The default router is "residual" (the
    # MLX-parity relevance scorer). The legacy multi-channel "SRL" router is
    # deprecated (net-negative in prior A/Bs) AND currently crashes on CUDA
    # decode with "tensors used as indices must be long, int, byte or bool
    # tensors" — the exception is swallowed, routing silently returns nothing,
    # and deep needles are lost. This test previously FORCED that broken path
    # via DIFFKV_SRL_THRESHOLD=5 and failed at 8000/0.1 with garbage output,
    # even though the residual router retrieves the needle (verified via
    # colab/diffkv_isolate.py --depth 0.1 --ctxs 8000). Pin the shipping router
    # and do NOT lower the SRL threshold (a low threshold builds the SRL index,
    # which is what triggers the crashing code path).
    os.environ["DIFFKV_ROUTER"] = "residual"
    os.environ.pop("DIFFKV_SRL_THRESHOLD", None)

    # Routed-budget parity with MLX. The router keeps K blocks; the routed token
    # budget is K * block_size. MLX uses block_size=256 so K=16 covers 4096
    # tokens, but the CUDA manager hardcodes block_size=64, so the default K=16
    # covers only 1024 — 4x less — and a deep needle's block falls outside the
    # top-16 by relevance. K=64 restores the same 64*64=4096 routed budget as MLX
    # and reliably includes the far needle. (A100-verified via
    # colab/diffkv_needle_diag.py: needle_in_routed=True, n_residuals=64,
    # gen='847291' at 8000/0.1, with the metadata-sync + exact-residual fixes.)
    os.environ["DIFFKV_TOPK_BLOCKS"] = "64"

    # rank=32 matches the wrapper default and DIFFKV_RSVD_MAX_RPROJ=32;
    # rank=16 is too low for 14B models with RANK_BOOST=off (loses digit blocks).
    wrapper = DiffKVHFWrapper(MODEL, config={"rank": 32}, device=device)
    
    needle = "The special code is 847291."
    question = "What is the special code? Answer in exactly the 6-digit code number."
    
    prompt = make_niah_prompt(wrapper.tokenizer, context_len, depth, needle, question)
    
    print(f"\nRunning NIAH test: context={context_len}, depth={depth}")
    
    response = wrapper.generate(
        prompt=prompt,
        max_new_tokens=16,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
    )
    
    print(f"Response: {response!r}")

    # Primary check: extract only the newly-generated tokens from internal state.
    # gen_ids is empty (→ gen_text='') when the model outputs only a stop token
    # (EOS / <|im_end|>) as its first prediction — a KV-recall regression signal.
    prompt_toks = len(wrapper.tokenizer.encode(prompt))
    sid = wrapper.active_session or "default"
    all_ids = wrapper._session_token_ids.get(sid, [])
    gen_ids = all_ids[prompt_toks:]
    gen_text = wrapper.tokenizer.decode(gen_ids, skip_special_tokens=True)

    print(f"Generated text only: {gen_text!r}")
    print(f"Total tokens in session: {len(all_ids)}, prompt tokens: {prompt_toks}, new tokens: {len(gen_ids)}")

    # Fallback: the response string (decoded from prompt+gen with skip_special_tokens)
    # will contain the answer even when it appears after the prompt text, as long as
    # the model generated more than just a stop token.
    needle_in_gen = "847291" in gen_text
    # Strip the prompt text from response to isolate the assistant turn.
    # response is tokenizer.decode(prompt_ids + gen_ids, skip_special_tokens=True),
    # so everything after the last occurrence of the question is the answer.
    response_tail = response
    _q_idx = response.rfind("What is the special code")
    if _q_idx >= 0:
        response_tail = response[_q_idx:]
    needle_in_response = "847291" in response_tail

    assert needle_in_gen or needle_in_response, (
        f"Failed to retrieve needle '847291' at context_len={context_len}, depth={depth}.\n"
        f"  gen_text (new tokens only, skip_special): {gen_text!r}\n"
        f"  response_tail (after question): {response_tail[:200]!r}\n"
        f"  Total session tokens: {len(all_ids)} (prompt={prompt_toks}, new={len(gen_ids)})\n"
        f"  Hint: if new_tokens==1 and gen_text=='' the model predicted EOS immediately —"
        f" this is a KV recall regression (try higher rank or re-enable DIFFKV_RANK_BOOST)."
    )
    wrapper.stop()
