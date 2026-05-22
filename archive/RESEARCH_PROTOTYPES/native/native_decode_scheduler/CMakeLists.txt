cmake_minimum_required(VERSION 3.18)
project(native_decode_scheduler CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG QUIET)

if(NOT pybind11_FOUND)
    # Fallback: use pip-installed pybind11
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE PYBIND11_RESULT
    )
    if(PYBIND11_RESULT EQUAL 0)
        list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
        find_package(pybind11 CONFIG REQUIRED)
    else()
        message(FATAL_ERROR "pybind11 not found. Install with: pip install pybind11")
    endif()
endif()

pybind11_add_module(native_decode_scheduler
    native_decode_scheduler.cpp
    bindings.cpp
)

target_include_directories(native_decode_scheduler PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Windows-specific settings
if(WIN32)
    target_compile_definitions(native_decode_scheduler PRIVATE _WIN32_WINNT=0x0601)
    target_compile_options(native_decode_scheduler PRIVATE /O2 /W3)
else()
    target_compile_options(native_decode_scheduler PRIVATE -O3 -Wall)
endif()

install(TARGETS native_decode_scheduler DESTINATION .)

message(STATUS "native_decode_scheduler: Python=${Python3_VERSION} pybind11=${pybind11_VERSION}")
