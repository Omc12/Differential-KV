"""
scratch/profile_prefill_breakdown.py

Profiles where the remaining 2853ms prefill time is spent.
Measures:
  1. Per-chunk model forward time
  2. Ingest_chunk time per layer
  3. GC + empty_cache time (post-loop)
  4. Token transfer time
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
import gc

# ── Setup (reuse from comprehensive benchmark) ──────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
CONFIG = {
    "mode": "fp16",
    "block_size": 64,
    "rank": 16,
    "micro_block_size": 16,
    "serving_mode": "balanced",
}

try:
    from transformers import BitsAndBytesConfig
    quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
except:
    quant_cfg = None

from serving.hf_diffkv_wrapper import DiffKVHFWrapper

print("Loading model...")
wrapper = DiffKVHFWrapper(MODEL_ID, CONFIG, device="cuda", quantization_config=quant_cfg, torch_dtype=torch.float16)
tokenizer = wrapper.tokenizer

# ── Build a 2540-token prompt ──────────────────────────────────────────────
LONG_PROMPT = """[INST] You are a helpful assistant.

The following is a detailed historical record of events:

The year was 2157. Humanity had achieved faster-than-light travel through the discovery of quantum folding technology developed by Dr. Elena Thorne at the Jupiter Research Institute. The Aegis expedition set out with Major Helen Vance commanding a crew of forty-seven specialists. Their mission: to establish contact with the Luminari, a species discovered orbiting Europa.

The Luminari communicated through bioluminescent patterns that corresponded to mathematical sequences. Dr. Thorne's linguistic algorithms decoded these patterns over eighteen months of careful observation. The breakthrough came when a Luminari elder named Ix'ath demonstrated the equations for clean fusion using their crystalline computing matrices.

Earth governments had secretly prepared contingency protocols for hostile contact, but none were needed. The Luminari offered their fusion equations freely, knowing that humanity's energy crisis would inevitably destabilize the solar system's political balance. Within a decade, clean fusion reactors powered every major city on Earth, Luna, and the Martian colonies.

The second expedition, launched in 2167, brought three hundred colonists to Europa's subsurface ocean. Working alongside the Luminari, they built the first interspecies settlement: New Luminara, a city of glass tubes threading through geothermal vents. Children born in New Luminara could speak both Human Standard and Bioluminescent Pattern Language, though the latter required specialized retinal implants.

Major Helen Vance retired at age seventy-two, having served as governor of New Luminara for fifteen years. Her memoirs, dictated in both languages, became required reading in every school across the solar system. The final chapter described the moment she understood that humanity's greatest achievement was not the technology it gained, but the realization that intelligence itself was the universe's most precious commodity.

Dr. Thorne, meanwhile, continued her research into higher-dimensional mathematics with Ix'ath's great-grandchild, Kyl'ath. Together they published seventeen papers on topological consciousness, proposing that awareness itself was a fundamental force, like gravity or electromagnetism. Their work remained controversial but inspired three generations of theoretical physicists.

The Luminari worlds eventually expanded beyond Europa to include Ganymede and Callisto. By 2200, the Jupiter system hosted twelve billion beings of six distinct species, all operating under the Jovian Compact, a legal framework co-written by Helen Vance and Ix'ath during those first difficult months of contact. The Compact's first article stated simply: Intelligence recognizes intelligence.

This is the abbreviated historical record. Based on this, answer the following question in exactly three words: Name the two moons colonized by the second expedition.

[/INST]"""

tokens = tokenizer(LONG_PROMPT, return_tensors="pt")
prompt_ids = tokens.input_ids[0].tolist()
print(f"Prompt length: {len(prompt_ids)} tokens")

# ── Instrument ingest_chunk calls ────────────────────────────────────────────
original_ingest = wrapper.manager._streaming_mgr.ingest_chunk
ingest_times = []
def timed_ingest(session_id, layer_idx, k, v):
    t0 = time.perf_counter()
    result = original_ingest(session_id, layer_idx, k, v)
    ingest_times.append((time.perf_counter() - t0) * 1000)
    return result
wrapper.manager._streaming_mgr.ingest_chunk = timed_ingest

# ── Run prefill ───────────────────────────────────────────────────────────────
session_id = "profile_test"
wrapper.model._diffkv_session_ids = [session_id]
wrapper.manager.init_session(session_id, prefill_len=len(prompt_ids))

chunk_size = 512
chunk_times = []
transfer_times = []
gc_time = 0.0

_input_buf = torch.zeros((1, chunk_size), dtype=torch.long).pin_memory()
L_new = len(prompt_ids)

t_total = time.perf_counter()
for offset in range(0, L_new, chunk_size):
    chunk_ids = prompt_ids[offset : offset + chunk_size]
    actual_len = len(chunk_ids)

    t_xfer = time.perf_counter()
    _input_buf[0, :actual_len] = torch.as_tensor(chunk_ids, dtype=torch.long)
    input_ids = _input_buf[:, :actual_len].to("cuda", non_blocking=True)
    position_ids = torch.arange(offset, offset + actual_len, dtype=torch.long, device="cuda").unsqueeze(0)
    torch.cuda.synchronize()
    transfer_times.append((time.perf_counter() - t_xfer) * 1000)

    t_fwd = time.perf_counter()
    with torch.no_grad():
        out = wrapper.model(input_ids=input_ids, position_ids=position_ids, use_cache=True)
    torch.cuda.synchronize()
    chunk_times.append((time.perf_counter() - t_fwd) * 1000)

t_gc = time.perf_counter()
del _input_buf
gc.collect()
torch.cuda.empty_cache()
gc_time = (time.perf_counter() - t_gc) * 1000
total_time = (time.perf_counter() - t_total) * 1000

print(f"\n{'='*60}")
print(f"PREFILL BREAKDOWN ({len(prompt_ids)} tokens, {len(chunk_times)} chunks)")
print(f"{'='*60}")
print(f"Total prefill time:    {total_time:.1f} ms")
print(f"\nPer-chunk forward times (ms):")
for i, ct in enumerate(chunk_times):
    print(f"  Chunk {i} ({chunk_size if i < len(chunk_times)-1 else L_new % chunk_size or chunk_size} tokens): {ct:.1f} ms")
print(f"\nTransfer overhead per chunk: {[f'{t:.1f}' for t in transfer_times]} ms")
print(f"Post-loop gc+empty_cache:   {gc_time:.1f} ms")
print(f"\nIngest calls: {len(ingest_times)}")
print(f"Total ingest time:   {sum(ingest_times):.1f} ms")
print(f"Avg ingest per call: {sum(ingest_times)/len(ingest_times):.3f} ms")
print(f"Max ingest per call: {max(ingest_times):.3f} ms")
print(f"\nForward pass total:  {sum(chunk_times):.1f} ms")
print(f"Ingest total:        {sum(ingest_times):.1f} ms")
print(f"Transfer total:      {sum(transfer_times):.1f} ms")
print(f"GC/cache total:      {gc_time:.1f} ms")
print(f"Unaccounted:         {total_time - sum(chunk_times) - sum(ingest_times) - sum(transfer_times) - gc_time:.1f} ms")
