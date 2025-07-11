# Matrix Multiplication Optimization Suite

A comprehensive collection of highly optimized matrix multiplication implementations for both CPU and GPU architectures, demonstrating advanced performance optimization techniques across different computational paradigms.

## Project Structure

```
matrixmult/
├── CPU/                          # CPU-based implementations
│   ├── dense-matmul/            # Dense matrix multiplication with SIMD/OpenMP
│   └── sparse-matmul/           # Sparse matrix multiplication using CSR format
├── GPU/                          # GPU-based CUDA implementations
│   ├── matmul/                  # GPU matrix multiplication kernels
│   └── histogram/               # Optimized histogram computation
└── README.md                    # This file
```

## Overview

This repository contains four distinct optimization projects, each targeting specific computational challenges:

### CPU Implementations
- **Dense Matrix Multiplication**: Highly optimized GEMM using AVX-512, cache blocking, and OpenMP
- **Sparse Matrix Multiplication**: CSR format implementation with multi-threading optimization

### GPU Implementations  
- **Matrix Multiplication**: CUDA kernels with shared memory optimization and memory coalescing
- **Histogram Computation**: Efficient parallel histogram calculation with warp-level optimizations

## Build System

Each subproject uses CMake for cross-platform building:

```bash
# Navigate to any subproject
cd CPU/dense-matmul

# Create build directory
mkdir build && cd build

# Configure and build
cmake ..
make
```

## Performance Features

### CPU Optimizations
- **SIMD Vectorization**: AVX-512 instructions for parallel processing
- **Cache Blocking**: Multi-level tiling for optimal memory hierarchy usage
- **OpenMP Parallelization**: Multi-core CPU utilization
- **Memory Alignment**: 64-byte alignment for optimal SIMD performance
- **Loop Unrolling**: Register blocking and instruction-level parallelism

### GPU Optimizations
- **Shared Memory**: Fast on-chip memory utilization
- **Memory Coalescing**: Optimized global memory access patterns
- **Warp-Level Primitives**: Efficient thread cooperation
- **Asynchronous Operations**: Overlapped computation and memory transfers
- **Pinned Memory**: Faster host-device transfers

## Dependencies

### CPU Projects
- **Compiler**: GCC with AVX-512 support or compatible
- **OpenMP**: For parallel processing
- **CMake**: Version 3.10 or higher

### GPU Projects
- **CUDA Toolkit**: Version 10.0 or higher
- **NVIDIA GPU**: Compute capability 6.0+
- **CMake**: Version 3.18 or higher

## Individual Project Documentation

Each subproject contains detailed documentation:

- [`CPU/dense-matmul/README.md`](CPU/dense-matmul/README.md) - Dense matrix multiplication optimizations
- [`CPU/sparse-matmul/README.md`](CPU/sparse-matmul/README.md) - Sparse matrix techniques and CSR format
- [`GPU/matmul/README.md`](GPU/matmul/README.md) - CUDA matrix multiplication kernels
- [`GPU/histogram/README.md`](GPU/histogram/README.md) - Parallel histogram computation

## Quick Start

### Running CPU Dense Matrix Multiplication
```bash
cd CPU/dense-matmul
mkdir build && cd build
cmake ..
make
./dense_matmul <matrix1_file> <matrix2_file> <n> <k> <m>
```

### Running GPU Matrix Multiplication
```bash
cd GPU/matmul
mkdir build && cd build
cmake ..
make
./gpu_matmul <matrix1_file> <matrix2_file> <n> <k> <m>
```

## Testing

Each project includes comprehensive testing:

```bash
# Build and run tests
cd <project_directory>/build
make
make test
```

## Performance Metrics

The implementations demonstrate significant performance improvements across all optimization stages:

### Performance Speedup Table

| Algorithm | Platform | Implementation Stage | Runtime | Speedup vs. Baseline |
|-----------|----------|---------------------|---------|----------------------|
| **Dense GEMM** | CPU | Naive Implementation | 13.0s | 1.0x (baseline) |
| | | Basic OpenMP | 3.2s | 4.1x |
| | | SIMD + Cache Blocking | 800ms | 16.3x |
| | | **Final Optimized** | **200ms** | **65.0x** |
| **Sparse SpMM** | CPU | Naive Implementation | 35.0s | 1.0x (baseline) |
| | | Basic CSR Format | 8.5s | 4.1x |
| | | OpenMP Parallelization | 2.1s | 16.7x |
| | | **Final Optimized** | **150ms** | **233.3x** |
| **Dense GEMM** | GPU | Naive CUDA | 30.0s | 1.0x (baseline) |
| | | Basic Tiling | 7.5s | 4.0x |
| | | Shared Memory Opt. | 1.8s | 16.7x |
| | | **Final Optimized** | **130ms** | **230.8x** |
| **Histogram** | GPU | Naive CUDA | 12.0s | 1.0x (baseline) |
| | | Basic Atomics | 4.2s | 2.9x |
| | | Privatized Approach | 1.5s | 8.0x |
| | | **Final Optimized** | **800ms** | **15.0x** |

### Key Performance Achievements

- **CPU Dense GEMM**: 65x speedup (13s → 200ms) through AVX-512 vectorization and cache optimization
- **CPU Sparse SpMM**: 233x speedup (35s → 150ms) with CSR format and parallel processing
- **GPU Matrix Multiplication**: 231x speedup (30s → 130ms) via shared memory and memory coalescing
- **GPU Histogram**: 15x speedup (12s → 800ms) using privatized histograms and warp-level optimizations

## Compilation Flags

### CPU Projects
```bash
-O3 -march=native -mavx512f -mavx512cd -fopenmp
```

### GPU Projects
```bash
-O3 -arch=sm_70 -lineinfo --use_fast_math
```

## Algorithm Highlights

### Dense GEMM (CPU)
- Three-level cache blocking strategy
- AVX-512 vectorization with FMA instructions
- Register blocking and loop unrolling
- Non-temporal stores for large matrices

### Sparse SpMM (CPU)
- CSR format for memory efficiency
- Dynamic OpenMP scheduling
- Parallel matrix format conversion
- Optimized for various sparsity patterns

### GPU Matrix Multiplication
- Tiled shared memory algorithm
- Thread block optimization
- Bank conflict avoidance
- Double buffering techniques

### GPU Histogram
- Privatized histogram approach
- Atomic operation optimization
- Warp-level reduction techniques
- Load balancing across SMs

## Contributing

Each project follows consistent patterns:
1. Core algorithm in `src/main.cpp` or `src/main.cu`
2. Interface defined in `include/studentlib.h`
3. Test suite in `tester/`
4. Build configuration in `CMakeLists.txt`

## License

This project is part of an academic optimization study demonstrating high-performance computing techniques across CPU and GPU architectures.

---

**Note**: Each subproject is self-contained and can be built independently. Refer to individual README files for detailed optimization explanations and performance analysis.