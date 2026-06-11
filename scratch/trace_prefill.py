import os
import sys

# Add root dir to path
sys.path.insert(0, "/Users/omchimurkar1/Desktop/Differential-KV/ACTIVE_RUNTIME")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

prompt = """Random Features for Large-Scale Kernel Machines Ali Rahimi and Ben Recht Abstract To accelerate the training of kernel machines, we propose to map the input data to a randomized low-dimensional feature space and then apply existing fast linear methods. Our randomized features are designed so that the inner products of the transformed data are approximately equal to those in the feature space of a user specified shift-invariant kernel. We explore two sets of random features, provide convergence bounds on their ability to approximate various radial basis kernels, and show that in large-scale classification and regression tasks linear machine learning algorithms that use these features outperform state-of-the-art large-scale kernel machines. 1 Introduction Kernel machines such as the Support Vector Machine are attractive because they can approximate any function or decision boundary arbitrarily well with enough training data. Unfortunately, methods that operate on the kernel matrix (Gram matrix) of the data scale poorly with the size of the training dataset. For example, a dataset with half a million training examples might take days to train on modern workstations. On the other hand, specialized algorithms for linear Support Vector Machines and regularized regression run much more quickly when the dimensionality of the data is small because they operate on the covariance matrix rather than the kernel matrix of the training data [1, 2]. We propose a way to combine the advantages of the linear and nonlinear approaches. Inspired by randomized algorithms for approximating kernel matrices (e.g., [3, 4]), we efficiently convert the training and evaluation of any kernel machine into the corresponding operations of a linear machine by mapping data into a relatively low-dimensional randomized feature space. Our experiments show that random features combined with very simple linear learning techniques compete favorably with state-of-the-art kernel-based classification and regression algorithms. Random features significantly reduce the computation needed for training, and obtain similar or better testing error. The kernel trick is a simple way to generate features for algorithms that depend only on the inner product between pairs of input points."""

# 1. Raw prompt tokenization
tokens_raw = tokenizer.encode(prompt)
print(f"Raw prompt tokens count: {len(tokens_raw)}")
print("First 20 raw tokens:")
for i, t in enumerate(tokens_raw[:20]):
    print(f"  {i}: {t} -> {repr(tokenizer.decode([t]))}")

# 2. Chat template tokenization
chat = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
chat_prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
tokens_chat = tokenizer.encode(chat_prompt)
print(f"\nChat prompt length: {len(chat_prompt)} chars, {len(tokens_chat)} tokens")
print("First 100 chat tokens:")
for i, t in enumerate(tokens_chat[:100]):
    piece = tokenizer.decode([t])
    print(f"  {i:3d}: token_id={t:6d} -> {repr(piece)}")

