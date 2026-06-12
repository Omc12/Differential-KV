import subprocess
import os
import time

def main():
    os.environ["DIFFKV_MPS_APPROXIMATE_ATTN"] = "0" # Exact attention
    os.environ["DIFFKV_ENGAGE_THRESHOLD"] = "0"
    os.environ["DIFFKV_IMMEDIATE_PREFILL_COMPRESS"] = "1"
    os.environ["DIFFKV_MAX_TOKENS"] = "128"
    
    binary_path = "/Users/omchimurkar1/Desktop/Differential-KV/diffkv_native/build/diffkv_native"
    model_path = "/Users/omchimurkar1/Desktop/Differential-KV/diffkv_native/qwen2.5-0.5b-instruct.gguf"
    
    # Large prompt
    paper_abstract = """Random Features for Large-Scale Kernel Machines Ali Rahimi and Ben Recht Abstract To accelerate the training of kernel machines, we propose to map the input data to a randomized low-dimensional feature space and then apply existing fast linear methods. Our randomized features are designed so that the inner products of the transformed data are approximately equal to those in the feature space of a user specified shift-invariant kernel. We explore two sets of random features, provide convergence bounds on their ability to approximate various radial basis kernels, and show that in large-scale classification and regression tasks linear machine learning algorithms that use these features outperform state-of-the-art large-scale kernel machines. 1 Introduction Kernel machines such as the Support Vector Machine are attractive because they can approximate an. """
    prompt = paper_abstract * 34 + "\n\nBased on the text above, summarize the key contributions and features in a detailed bulleted list:"
    
    chat_prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n" + prompt + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    # Escape newlines
    single_line = chat_prompt.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "")
    
    print("Launching C++ process in interactive mode...")
    proc = subprocess.Popen([binary_path, model_path, "-"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Wait for ready
    print("Waiting for __READY__...")
    ready_line = ""
    while "__READY__" not in ready_line:
        ready_line = proc.stdout.readline()
        print(f"Subprocess stdout: {ready_line.strip()}")
    
    print("Writing prompt...")
    proc.stdin.write(single_line + "\n")
    proc.stdin.flush()
    
    # Now read stdout
    print("Reading response...")
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        print(f"Stdout: {line.strip()}")
        if "__FINISH__" in line:
            break
            
    stderr = proc.stderr.read()
    print("\n--- STDERR ---")
    print(stderr)

if __name__ == "__main__":
    main()
