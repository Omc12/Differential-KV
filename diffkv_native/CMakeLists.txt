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
    native_core/srl/factual_store.cpp
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
    list(APPEND SOURCES
        native_core/diffkv_core/src/paging_stream.cu
        native_core/diffkv_core/src/diffkv_decode.cu
    )
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
    # Portable fallback path to the Metal shader source, pointing at THIS build's
    # source tree (not any author-specific home dir). Overridable at runtime with
    # the DIFFKV_METAL_DIR env var. Used by runtime/diffkv_attention.mm.
    target_compile_definitions(diffkv_native PRIVATE
        "DIFFKV_METAL_SOURCE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/native_core/diffkv_core/metal\"")
endif()

if (GGML_CUDA)
    target_link_libraries(diffkv_native PRIVATE
        CUDA::cusolver
        CUDA::cublas
        CUDA::cudart
    )
endif()

# Standalone conformance test executable
set(CONF_SOURCES
    ../tools/conformance_test.cpp
    runtime/native_block_pool.cpp
    runtime/diffkv_attention.cpp
)
if (APPLE)
    list(APPEND CONF_SOURCES runtime/diffkv_attention.mm)
endif()

add_executable(conformance_test ${CONF_SOURCES})
target_link_libraries(conformance_test PRIVATE
    llama
    ggml
)
if (APPLE)
    target_link_libraries(conformance_test PRIVATE
        ${FOUNDATION_LIBRARY}
        ${METAL_LIBRARY}
        ${ACCELERATE_LIBRARY}
    )
    # Enable Objective-C++ compilation flags for Metal files in conformance_test
    set_source_files_properties(runtime/diffkv_attention.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
endif()

