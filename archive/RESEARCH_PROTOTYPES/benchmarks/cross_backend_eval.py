import time
from compiler.manifold_graph_builder import ManifoldGraphBuilder
from compiler.cuda_backend import CUDABackend
from compiler.triton_backend import TritonBackend
from compiler.metal_backend import MetalBackend

def evaluate_backends():
    print("=== UCC Cross-Backend Evaluation ===")
    config = {"num_layers": 32, "num_heads": 32, "model_name": "llama3_8b"}
    builder = ManifoldGraphBuilder(config)
    for i in range(32):
        builder.build_transformer_block(i)
    graph = builder.finalize()
    
    backends = {
        "CUDA": CUDABackend(),
        "Triton": TritonBackend(),
        "Metal": MetalBackend()
    }
    
    for name, backend in backends.items():
        start = time.time()
        backend.lower_graph(graph)
        print(f"Backend: {name}, Lowering latency: {(time.time()-start)*1000:.4f}ms")

if __name__ == "__main__":
    evaluate_backends()
