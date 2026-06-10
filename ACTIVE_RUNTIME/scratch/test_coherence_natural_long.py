import os
import sys
import torch
import asyncio

# Ensure ACTIVE_RUNTIME is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from serving.hf_diffkv_wrapper import DiffKVHFWrapper
from serving.batch_engine import ContinuousBatchEngine

chapters = [
    "### Chapter 1: The Colonization of Europa\n"
    "In the mid-22nd century, humankind embarked on its most ambitious outer-system project: the colonization "
    "of Europa, Jupiter's icy moon. Initially thought to be a barren sheet of ice, advanced thermal probes "
    "revealed a massive sub-surface ocean warmed by hydrothermal vents. Under the leadership of the United Nations "
    "Space Administration (UNSA), dome-enclosed cities were built directly onto the icy crust. These cities, "
    "known as the Hydro-domes, housed thousands of scientists, deep-sea engineers, and miners. They harvested "
    "deuterium and heavy elements from the sea floor to power the solar system's fusion grid. Life on Europa was "
    "harsh but prosperous. The sub-crust oceans offered endless potential, yet they remained mostly unexplored "
    "due to intense water pressure and freezing rifts. The mystery of the deep Jovian sea continued to intrigue "
    "explorers and visionaries worldwide.\n\n",
    
    "### Chapter 2: The Aegis Expedition\n"
    "In 2384, a high-priority mission was authorized to explore the Mariana-class thermal rifts near Europa's equator. "
    "Major Helen Vance, an decorated deep-ocean navigator and expert in sub-glacial acoustics, was chosen to lead the "
    "expedition. Her vehicle was the Aegis, a state-of-the-art submarine engineered from reinforced carbon-nanotubes "
    "capable of withstanding the crushing depths of 100 kilometers below the ice sheet. Joining her was Dr. Aris Thorne, "
    "a specialist in marine exobiology. The Aegis descended through the primary ice-shaft into the freezing, dark "
    "waters. Guided only by sonar and high-powered searchlights, the crew traveled deeper than any human vehicle had "
    "ever ventured. Their objective was to investigate thermal anomalies that suggested geological activity, but "
    "what they found would completely redefine humanity's place in the cosmos.\n\n",
    
    "### Chapter 3: Bioluminescent Encounters\n"
    "As the Aegis approached the thermal rift, the crew noticed faint, pulsing lights in the distance. The lights "
    "did not originate from volcanic vents, but from living organisms. They discovered a highly intelligent glowing "
    "aquatic species, which Dr. Thorne named the Luminari. These creatures possessed sleek, transparent bodies with "
    "internal nodes that pulsed with vibrant light. Helen realized that the light sequences were not random; they "
    "conformed to highly structured mathematical sequences. By utilizing the Aegis's external bioluminescent arrays, "
    "Helen responded to the patterns. The Luminari danced in geometric formations, establishing a complex "
    "two-way communication channel. Dr. Thorne recorded sub-oceanic sound waves that accompanied the light rifts, "
    "marking the first official contact with intelligent extraterrestrial life in human history.\n\n",
    
    "### Chapter 4: The Clean Fusion Breakthrough\n"
    "Over several weeks of interaction, the Aegis team recorded thousands of light sequences. When analyzed by the "
    "supercomputers back at the Hydro-domes, the results were astonishing. The Luminari were conveying complex "
    "quantum equations. These equations described a method for stabilizing high-temperature plasma using magnetic fields "
    "inspired by their own bioluminescent node structures. This breakthrough resolved the centuries-old plasma containment "
    "issue, paving the way for a new class of clean fusion energy. This clean fusion technology offered virtually "
    "unlimited energy without any radioactive waste. Helen Vance became a global hero, and the equations provided "
    "by the Luminari were immediately implemented in Europa's primary energy grid before being sent back to Earth.\n\n",
    
    "### Chapter 5: Dawn of the Interstellar Era\n"
    "The clean fusion breakthrough revolutionized space exploration. With unlimited energy and incredibly efficient "
    "containment fields, interstellar travel became a realistic goal. Humankind constructed the first fusion-powered "
    "starships, enabling journeys beyond the solar system. Europa transformed from a resource mining outpost into "
    "the primary shipyard for the interstellar fleet. The Luminari remained peaceful partners, and Helen Vance's "
    "expedition was remembered as the pivotal moment that unlocked the stars for humanity. The Aegis was retired "
    "to the Central Museum of Europa, a symbol of curiosity, courage, and first contact.\n\n"
]

long_document = "".join(chapters) * 3

async def run_diffkv(prompt, preset="low", rank=16):
    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    os.environ["DIFFKV_TELEMETRY"] = "1"
    os.environ["DIFFKV_MPS_APPROXIMATE_ATTN"] = "1"
    
    print(f"\n--- Running DiffKV (Preset={preset}, Rank={rank}) ---")
    wrapper = DiffKVHFWrapper(MODEL, config={"preset": preset, "rank": rank}, device=device)
    engine = ContinuousBatchEngine(wrapper, max_batch_size=1)
    engine.start()
    
    q = await engine.submit("sess_diffkv", {
        "prompt": prompt,
        "max_tokens": 100,
        "temperature": 0.0,
    })
    
    full_output = []
    while True:
        chunk = await q.get()
        if "error" in chunk:
            print(f"Error: {chunk['error']}")
            break
        text = chunk.get("text", "")
        if text:
            full_output.append(text)
        if chunk.get("is_final"):
            break
            
    ans = "".join(full_output).strip()
    await engine.stop()
    wrapper.close()
    
    import gc
    del wrapper, engine
    gc.collect()
    torch.mps.empty_cache()
    return ans

async def run_dense(prompt):
    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    print("\n--- Running Standard HF (Dense) ---")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map=device
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=100, temperature=0.0, do_sample=False)
        
    generated = out[0][inputs.input_ids.shape[1]:]
    ans = tokenizer.decode(generated, skip_special_tokens=True).strip()
    
    import gc
    del model, tokenizer
    gc.collect()
    torch.mps.empty_cache()
    return ans

async def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Construct prompt with header and footer
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nHere is the historical record:\n"
        + long_document +
        "\n\nQuestion: Based on the provided historical record, who was the submarine major and what was the submarine named? Answer in one concise sentence.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    tokens = len(tokenizer.encode(prompt))
    print(f"Total prompt length: {tokens} tokens")
    
    dense_ans = await run_dense(prompt)
    print(f"\n======================================")
    print(f"DENSE OUTPUT:")
    print(f"======================================")
    print(dense_ans)
    
    diffkv_ans = await run_diffkv(prompt, preset="low", rank=16)
    print(f"\n======================================")
    print(f"DIFFKV OUTPUT:")
    print(f"======================================")
    print(diffkv_ans)

if __name__ == "__main__":
    asyncio.run(main())
