import os
import sys
import torch
import asyncio

# Ensure ACTIVE_RUNTIME is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from serving.hf_diffkv_wrapper import DiffKVHFWrapper
from serving.batch_engine import ContinuousBatchEngine

# Construct a unique, non-repetitive 8,000+ token document by combining different topics.
# Each topic is completely unique, avoiding self-attention repetition collapse.
topics = [
    # Topic 1: History of Computing
    "The history of computing hardware covers the developments from early simple devices to aid calculation "
    "to the modern microprocessors. The abacus was early used for arithmetic tasks. The slide rule was invented "
    "in the 17th century by William Oughtred and was widely used until the mid-20th century when electronic calculators "
    "became common. The first programmable computer was designed by Charles Babbage in the 1830s, known as the "
    "Analytical Engine. Although it was never fully completed during his lifetime due to funding issues and "
    "manufacturing limitations, Babbage's design laid the logical foundations for modern electronic computers, "
    "including a memory unit called the store and a processing unit called the mill. Ada Lovelace wrote the first "
    "algorithm intended to be processed by Babbage's machine, making her the world's first computer programmer.\n\n",
    
    # Topic 2: Europa Moon Exploration
    "Europa, Jupiter's icy moon, has long been a subject of fascination for planetary scientists. Under its frozen "
    "surface lies a vast ocean containing more water than all of Earth's oceans combined. Planetary missions such "
    "as Galileo and Juno have provided strong evidence of this sub-glacial ocean by measuring Europa's magnetic field "
    "variations as it interacts with Jupiter's magnetosphere. Scientists believe that hydrothermal vents on Europa's "
    "ocean floor could provide the heat and chemical energy necessary to support microbial life, similar to extreme "
    "ecosystems found in Earth's deep ocean trenches. Future space missions, including NASA's Europa Clipper and "
    "ESA's Jupiter Icy Moons Explorer (JUICE), aim to analyze the moon's ice shell thickness and search for organic "
    "compounds in the plumes of water vapor that occasionally erupt from the surface.\n\n",
    
    # Topic 3: Quantum Mechanics and Entanglement
    "Quantum mechanics is a fundamental theory in physics that describes the physical properties of nature at the scale "
    "of atoms and subatomic particles. Unlike classical physics, which assumes determinism, quantum mechanics introduces "
    "probabilistic outcomes. One of the most famous phenomena in this field is quantum entanglement, where two or "
    "more particles become interconnected in such a way that the state of one instantly determines the state of the other, "
    "regardless of the distance separating them. Albert Einstein famously referred to this phenomenon as 'spooky action "
    "at a distance' because it seemed to violate the principle of local realism. Today, quantum entanglement is the "
    "cornerstone of quantum computing and quantum cryptography, enabling secure communication and exponentially faster "
    "computational speeds for certain algorithmic tasks.\n\n",
    
    # Topic 4: Machine Learning and Neural Networks
    "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data "
    "and improve their performance over time without being explicitly programmed. Deep learning, which uses multi-layered "
    "artificial neural networks, has revolutionized fields such as computer vision, natural language processing, and "
    "speech recognition. These neural networks are inspired by the structure of the human brain, consisting of nodes "
    "and weighted connections that adjust during training using algorithms like backpropagation. Although neural networks "
    "were conceptualized in the mid-20th century, their recent success is driven by the availability of massive datasets "
    "and high-performance graphical processing units (GPUs) capable of processing millions of matrix multiplications "
    "simultaneously.\n\n",
    
    # Topic 5: Fusion Energy and Tokamaks
    "Nuclear fusion is the process that powers the sun and other stars, where light atomic nuclei combine to form heavier "
    "nuclei, releasing vast amounts of energy. Replicating this process on Earth has been a goal of energy researchers "
    "for decades, as fusion offers a virtually limitless supply of clean energy with no long-lived radioactive waste. "
    "The most common device used to contain fusion plasma is the tokamak, a torus-shaped chamber that uses strong magnetic "
    "fields to trap and heat isotopes of hydrogen, namely deuterium and tritium. The international ITER project in France "
    "is currently the largest fusion experiment, aiming to demonstrate a net energy gain, where the fusion power produced "
    "exceeds the external heating power injected into the plasma.\n\n",

    # Topic 6: The History of Writing
    "The history of writing systems describes the development of expressing language by letters or other marks and "
    "also the study and description of these developments. In the History of how writing systems have evolved in "
    "different human civilizations, more complete writing systems were preceded by proto-writing, systems of ideographic "
    "or early mnemonic symbols. True writing, in which the content of a linguistic utterance is encoded so that another "
    "reader can reconstruct, with a fair degree of accuracy, the exact utterance written down, is a later development. "
    "It is distinguished from proto-writing, which typically avoids encoding grammatical words and affixes, making it "
    "more difficult or impossible to reconstruct the exact meaning intended by the writer. Cuneiform, developed by the "
    "Sumerians in Mesopotamia around 3200 BC, is widely considered one of the earliest systems of writing.\n\n",

    # Topic 7: Photosynthesis and Plant Biology
    "Photosynthesis is a process used by plants, algae, and certain bacteria to harness energy from sunlight and "
    "turn it into chemical energy. In plants, this process occurs in organelles called chloroplasts, which contain "
    "the pigment chlorophyll. Chlorophyll absorbs light energy, primarily in the blue and red wavelengths, while "
    "reflecting green light, giving plants their characteristic color. The chemical energy produced is stored in "
    "carbohydrate molecules, such as sugars, which are synthesized from carbon dioxide and water. Photosynthesis is "
    "critical for maintaining life on Earth, as it is the primary source of organic material and oxygen in the "
    "atmosphere, driving the global carbon cycle and supporting terrestrial food webs.\n\n",

    # Topic 8: Plate Tectonics and Earth Geology
    "Plate tectonics is the scientific theory that explains the large-scale motions of Earth's lithosphere, which is "
    "divided into several tectonic plates. These plates move slowly over the ductile asthenosphere below them, driven "
    "by mantle convection and gravitational forces. The boundaries where tectonic plates meet are geologically active "
    "zones, associated with earthquakes, volcanic activity, mountain building, and oceanic trench formation. There are "
    "three primary types of plate boundaries: divergent boundaries where plates move apart, convergent boundaries "
    "where plates collide (often causing one plate to subduct beneath the other), and transform boundaries where plates "
    "slide past each other horizontally, such as the San Andreas Fault in California.\n\n"
]

# We will pad this uniquely to reach exactly 8,000+ tokens by repeating sections but with minor modifications,
# or we can repeat the whole block a few times. Let's repeat it 4 times.
# 4 times of these 8 long unique topics will yield ~8,000 tokens.
long_document = "".join(topics) * 4

async def run_diffkv(prompt, preset="low", rank=16):
    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    os.environ["DIFFKV_TELEMETRY"] = "1"
    os.environ["DIFFKV_MPS_APPROXIMATE_ATTN"] = "0"
    
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
        "\n\nQuestion: Based on the provided historical record, who wrote the first algorithm for Charles Babbage's Analytical Engine? Answer in one concise sentence.<|im_end|>\n"
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
