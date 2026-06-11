cmake_minimum_required(VERSION 3.20)
project(diffkv_native LANGUAGES CXX C OBJCXX OBJC)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Configuration options for llama.cpp / GGML
option(GGML_METAL "GGML: use Metal" ON)  # Enabled by default on Mac
option(GGML_CUDA "GGML: use CUDA" OFF)
option(LLAMA_BUILD_SERVER "llama: build server" OFF)

if (GGML_CUDA)
    enable_language(CUDA)
    # Ensure CUDA compiler flags are configured correctly
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    add_compile_definitions(GGML_USE_CUDA)
    add_compile_definitions(USE_CUDA)
endif()

# Add llama.cpp project
add_subdirectory(third_party/llama.cpp EXCLUDE_FROM_ALL)

# Header include paths
include_directories(
    .
    serving
    native_core
    native_core/diffkv_core/include
    third_party/llama.cpp/include
    third_party/llama.cpp/ggml/include
    third_party/llama.cpp/ggml/src
)

# Source files for diffkv_native
set(SOURCES
    runtime/diffkv_model.cpp
    runtime/native_block_pool.cpp
    native_core/compression/async_compressor.cpp
    native_core/compression/lowrank.cpp
    native_core/srl/diffkv_srl.cpp
    native_core/paging/paged_kv_store.cpp
    native_core/streaming_sparse_ingest.cpp
    native_core/kv_runtime_manager.cpp
    runtime/diffkv_attention.cpp
    native_core/srl/query_router.cpp
    native_core/diffkv_core/src/bindings.cpp
    native_core/diffkv_core/src/srl_router.cpp
    native_core/diffkv_core/src/decode_attention.cpp
    native_core/diffkv_core/src/compressor_thread.cpp
    native_core/diffkv_core/src/compressor_thread_cpu.cpp
    native_core/diffkv_core/src/paging_stream.cpp
    native_core/graph_runtime/static_decode_graph.cpp
    serving/production_session_manager.cpp
    serving/batch_engine.cpp
    serving/openai_compatible_api_gateway.cpp
    src/main.cpp
)

# If compiling on macOS, add Metal files
if (APPLE)
    list(APPEND SOURCES runtime/diffkv_attention.mm native_core/diffkv_core/src/metal_runtime.mm)
    # Enable Objective-C++ compilation flags for Metal
    set_source_files_properties(runtime/diffkv_attention.mm native_core/diffkv_core/src/metal_runtime.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
    find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
    find_library(METAL_LIBRARY Metal REQUIRED)
    find_library(ACCELERATE_LIBRARY Accelerate REQUIRED)
endif()

if (GGML_CUDA)
    list(APPEND SOURCES native_core/diffkv_core/src/paging_stream.cu)
    find_package(CUDAToolkit REQUIRED)
endif()

# Build the executable
add_executable(diffkv_native ${SOURCES})

# Link libraries
target_link_libraries(diffkv_native PRIVATE
    llama
    ggml
)

if (APPLE)
    target_link_libraries(diffkv_native PRIVATE
        ${FOUNDATION_LIBRARY}
        ${METAL_LIBRARY}
        ${ACCELERATE_LIBRARY}
    )
endif()

if (GGML_CUDA)
    target_link_libraries(diffkv_native PRIVATE
        CUDA::cusolver
        CUDA::cublas
        CUDA::cudart
    )
endif()
