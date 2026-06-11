import subprocess
import os
import sys

binary_path = "/Users/omchimurkar1/Desktop/Differential-KV/diffkv_native/build/diffkv_native"
model_path = "/Users/omchimurkar1/Desktop/Differential-KV/diffkv_native/qwen2.5-0.5b-instruct.gguf"

# Escape prompt
prompt_content = "Random Features for Large-Scale Kernel Machines Ali Rahimi and Ben Recht Abstract To accelerate the training of kernel machines, we propose to map the input data to a randomized low-dimensional feature space and then apply existing fast linear methods. Our randomized features are designed so that the inner products of the transformed data are approximately equal to those in the feature space of a user specified shift-invariant kernel. We explore two sets of random features, provide convergence bounds on their ability to approximate various radial basis kernels, and show that in large-scale classification and regression tasks linear machine learning algorithms that use these features outperform state-of-the-art large-scale kernel machines. 1 Introduction Kernel machines such as the Support Vector Machine are attractive because they can approximate any function or decision boundary arbitrarily well with enough training data. Unfortunately, methods that operate on the kernel matrix (Gram matrix) of the data scale poorly with the size of the training dataset. For example, a dataset with half a million training examples might take days to train on modern workstations. On the other hand, specialized algorithms for linear Support Vector Machines and regularized regression run much more quickly when the dimensionality of the data is small because they operate on the covariance matrix rather than the kernel matrix of the training data [1, 2]. We propose a way to combine the advantages of the linear and nonlinear approaches. Inspired by randomized algorithms for approximating kernel matrices (e.g., [3, 4]), we efficiently convert the training and evaluation of any kernel machine into the corresponding operations of a linear machine by mapping data into a relatively low-dimensional randomized feature space. Our experiments show that random features combined with very simple linear learning techniques compete favorably with state-of-the-art kernel-based classification and regression algorithms. Random features significantly reduce the computation needed for training, and obtain similar or better testing error. The kernel trick is a simple way to generate features for algorithms that depend only on the inner product between pairs of input points."

full_escaped_prompt = f"<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n<|im_start|>user\\n{prompt_content}<|im_end|>\\n<|im_start|>assistant\\n"

print("Starting diffkv_native subprocess...")
process = subprocess.Popen(
    [binary_path, model_path, "-"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=None,  # Inherit stderr so it prints directly without blocking
    text=True,
    bufsize=1,
)

print("Waiting for __READY__...")
while True:
    line = process.stdout.readline()
    if not line:
        break
    print(f"STDOUT: {line.strip()}")
    if "__READY__" in line:
        break

print("Sending prompt...")
process.stdin.write("__CACHED__:0\n")
process.stdin.write(full_escaped_prompt + "\n")
process.stdin.flush()

print("Reading response...")
while True:
    line = process.stdout.readline()
    if not line:
        break
    print(f"STDOUT: {line.strip()}")
    if "__FINISH__" in line:
        break

# Read remainder of stdout
stdout_rem, _ = process.communicate(timeout=5)
print("\n--- STDOUT REMAINDER ---")
print(stdout_rem)
