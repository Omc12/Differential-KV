cmake_minimum_required(VERSION 3.20)
project(diffkv_native LANGUAGES CXX C OBJCXX OBJC)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Configuration options for llama.cpp / GGML
option(GGML_METAL "GGML: use Metal" ON)  # Enabled by default on Mac
option(LLAMA_BUILD_SERVER "llama: build server" OFF)

# Add llama.cpp project
add_subdirectory(third_party/llama.cpp EXCLUDE_FROM_ALL)

# Header include paths
include_directories(
    include
    third_party/llama.cpp/include
    third_party/llama.cpp/ggml/include
    third_party/llama.cpp/ggml/src
)

# Source files for diffkv_native
set(SOURCES
    src/diffkv_model.cpp
    src/diffkv_kv_engine.cpp
    src/diffkv_compressor.cpp
    src/diffkv_srl.cpp
    src/diffkv_attention.cpp
    src/main.cpp
)

# If compiling on macOS, add Metal files
if (APPLE)
    list(APPEND SOURCES src/diffkv_attention.mm)
    # Enable Objective-C++ compilation flags for Metal
    set_source_files_properties(src/diffkv_attention.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
    find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
    find_library(METAL_LIBRARY Metal REQUIRED)
    find_library(ACCELERATE_LIBRARY Accelerate REQUIRED)
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
