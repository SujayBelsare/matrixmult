cmake_minimum_required(VERSION 3.15)
project(MatrixMultiply LANGUAGES CXX)

# Library target
add_library(studentlib main.cpp)

# Include headers
target_include_directories(studentlib PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Compile options: full optimizations + AVX-512 + disable FMA contraction + disable loop unrolling
target_compile_options(studentlib PRIVATE
    -O3
    -march=skylake-avx512    # or -march=native on supported hardware
    -ffp-contract=off        # disable fused multiply-add
    -fno-unroll-loops        # disable loop unrolling
)

# Find and link OpenMP if available
find_package(OpenMP)
if (OpenMP_FOUND)
    target_compile_options(studentlib PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(studentlib PUBLIC OpenMP::OpenMP_CXX)
endif()

# Example: set default C++ standard
set_target_properties(studentlib PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)