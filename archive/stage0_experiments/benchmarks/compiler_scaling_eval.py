import time
import torch
from compiler.manifold_graph_builder import ManifoldGraphBuilder
from compiler.cuda_backend import CUDABackend

def benchmark_compile_time():
    print("=== UCC Compiler Scaling Evaluation ===")
    results = []
    
    for layers in [12, 24, 32, 80]:
        config = {"num_layers": layers, "num_heads": 32, "model_name": f"llama_{layers}l"}
        
        start = time.time()
        builder = ManifoldGraphBuilder(config)
        for i in range(layers):
            builder.build_transformer_block(i)
        graph = builder.finalize()
        
        backend = CUDABackend()
        backend.lower_graph(graph)
        
        end = time.time()
        compile_time = (end - start) * 1000
        print(f"Layers: {layers}, Compile Time: {compile_time:.2f}ms")
        results.append({"layers": layers, "time": compile_time})
        
    return results

if __name__ == "__main__":
    benchmark_compile_time()
