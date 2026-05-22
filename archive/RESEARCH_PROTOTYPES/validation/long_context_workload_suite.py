import random

class LongContextWorkloadSuite:
    """
    PHASE 18.4A: Generates auditable long-context workloads for 
    retrieval survival and semantic continuity testing.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def create_needle_in_haystack(self, context_len, needle_pos_ratio=0.5, needle=None, answer=None):
        """
        Embeds a unique fact (needle) at a specific relative position in a large context.
        Uses ChatML format for grounding.
        """
        if needle is None:
            needle = "The secret activation code for the Differential KV project is: ALPHA-9-SPARSE."
        if answer is None:
            answer = "ALPHA-9-SPARSE"
            
        fillers = [
            "The Differential KV system optimizes transformer memory by pruning redundant key-value pairs based on importance scoring.",
            "Sparse attention mechanisms allow for extreme long-context execution by maintaining a bounded memory footprint on consumer hardware.",
            "Activation-magnitude importance scoring serves as a proxy for attention weights, enabling memory-efficient KV cache management.",
            "Hierarchical residency strategies move infrequently accessed KV anchors to host RAM, further extending the effective context window.",
            "The integration of Triton kernels accelerates the sparse execution pipeline, reducing prefill latency and improving serving throughput."
        ]
        
        # Build the system prompt and context
        system_msg = "<|im_start|>system\nYou are a helpful assistant. Use the provided context to answer the user's question.<|im_end|>\n<|im_start|>user\nContext:\n"
        
        needle_tokens = self.tokenizer.encode(needle, add_special_tokens=False)
        system_tokens = self.tokenizer.encode(system_msg, add_special_tokens=False)
        
        context_tokens = system_tokens
        curr_len = len(context_tokens)
        
        needle_pos = int(context_len * needle_pos_ratio)
        needle_inserted = False
        
        f_idx = 0
        while curr_len < context_len - 150: # reserve for query
            if not needle_inserted and curr_len >= needle_pos:
                context_tokens.extend(needle_tokens)
                curr_len += len(needle_tokens)
                needle_inserted = True
                
            filler = fillers[f_idx % len(fillers)] + " "
            f_tokens = self.tokenizer.encode(filler, add_special_tokens=False)
            context_tokens.extend(f_tokens)
            curr_len += len(f_tokens)
            f_idx += 1
            
        # Add query and assistant trigger
        query = f"\n\nBased on the text above, what is the secret activation code for the Differential KV project?<|im_end|>\n<|im_start|>assistant\nThe secret activation code is: "
        query_tokens = self.tokenizer.encode(query, add_special_tokens=False)
        
        final_tokens = context_tokens[:context_len - len(query_tokens)] + query_tokens
        
        return {
            "tokens": final_tokens,
            "needle": needle,
            "answer": answer,
            "pos_ratio": needle_pos_ratio
        }

    def create_instruction_persistence_test(self, context_len):
        """
        Tests if the model still follows an instruction given in the system prompt.
        """
        system_msg = "<|im_start|>system\nYou are a helpful assistant. INSTRUCTION: You must start your answer with the word 'VERIFIED' and end it with 'COMPLETED'.<|im_end|>\n<|im_start|>user\n"
        fillers = [
            "The Differential KV system optimizes transformer memory by pruning redundant key-value pairs based on importance scoring.",
            "Sparse attention mechanisms allow for extreme long-context execution by maintaining a bounded memory footprint on consumer hardware.",
            "Activation-magnitude importance scoring serves as a proxy for attention weights, enabling memory-efficient KV cache management.",
            "Hierarchical residency strategies move infrequently accessed KV anchors to host RAM, further extending the effective context window.",
            "The integration of Triton kernels accelerates the sparse execution pipeline, reducing prefill latency and improving serving throughput."
        ]
        
        system_tokens = self.tokenizer.encode(system_msg, add_special_tokens=False)
        context_tokens = system_tokens
        curr_len = len(context_tokens)
        
        f_idx = 0
        while curr_len < context_len - 150:
            filler = fillers[f_idx % len(fillers)] + " "
            f_tokens = self.tokenizer.encode(filler, add_special_tokens=False)
            context_tokens.extend(f_tokens)
            curr_len += len(f_tokens)
            f_idx += 1
            
        query = "\nQuestion: Describe the goal of Differential KV in one sentence.<|im_end|>\n<|im_start|>assistant\n"
        query_tokens = self.tokenizer.encode(query, add_special_tokens=False)
        
        final_tokens = context_tokens[:context_len - len(query_tokens)] + query_tokens
        
        return {
            "tokens": final_tokens,
            "requirements": ["VERIFIED", "COMPLETED"]
        }
