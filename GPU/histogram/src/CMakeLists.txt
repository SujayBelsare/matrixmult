cmake_minimum_required(VERSION 3.10)
project(studentlib)

# Enable CUDA
find_package(CUDA REQUIRED)

# Add CUDA compilation flags
set(CUDA_NVCC_FLAGS 
    ${CUDA_NVCC_FLAGS} 
    -O3 
    -Xcompiler -march=native
)

# Create the library
cuda_add_library(studentlib main.cu)
target_include_directories(studentlib PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Set additional compiler flags for host code
target_compile_options(studentlib PRIVATE -Ofast -march=native)