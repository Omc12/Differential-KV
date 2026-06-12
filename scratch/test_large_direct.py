import subprocess
import os

def main():
    # Set environment variables for exact or approximate attention
    os.environ["DIFFKV_MPS_APPROXIMATE_ATTN"] = "0" # Exact attention first
    os.environ["DIFFKV_ENGAGE_THRESHOLD"] = "0"
    os.environ["DIFFKV_IMMEDIATE_PREFILL_COMPRESS"] = "1"
    os.environ["DIFFKV_MAX_TOKENS"] = "128"
    
    binary_path = "/Users/omchimurkar1/Desktop/Differential-KV/diffkv_native/build/diffkv_native"
    model_path = "/Users/omchimurkar1/Desktop/Differential-KV/diffkv_native/qwen2.5-0.5b-instruct.gguf"
    
    paper_abstract = """Random Features for Large-Scale Kernel Machines Ali Rahimi and Ben Recht Abstract To accelerate the training of kernel machines, we propose to map the input data to a randomized low-dimensional feature space and then apply existing fast linear methods. Our randomized features are designed so that the inner products of the transformed data are approximately equal to those in the feature space of a user specified shift-invariant kernel. We explore two sets of random features, provide convergence bounds on their ability to approximate various radial basis kernels, and show that in large-scale classification and regression tasks linear machine learning algorithms that use these features outperform state-of-the-art large-scale kernel machines. 1 Introduction Kernel machines such as the Support Vector Machine are attractive because they can approximate an. """
    prompt = paper_abstract * 34 + "\n\nBased on the text above, summarize the key contributions and features in a detailed bulleted list:"
    
    chat_prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n" + prompt + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    print("Running C++ diffkv_native directly with large prompt...")
    cmd = [binary_path, model_path, chat_prompt]
    
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = proc.communicate()
    
    print("\n--- STDERR ---")
    print(stderr)
    print("\n--- STDOUT ---")
    print(stdout)

if __name__ == "__main__":
    main()
