def get_workloads():
    return [
        {"name": "4k Chat Serving", "context": 4096, "prompts": ["Explain quantum entanglement in simple terms.", "Write a python script to scrape a website."]},
        {"name": "16k Retrieval-Heavy", "context": 16384, "prompts": ["Summarize the attached 16k context document.", "Find the key architecture decisions in this codebases metadata."]},
        {"name": "32k Sparse Memory", "context": 32768, "prompts": ["Retrieve the specific versioning logic from the early project logs.", "Explain the evolution of the sparse paging engine."]}
    ]
