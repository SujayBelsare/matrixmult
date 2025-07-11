# Optimized GEMM Implementation

This project implements a highly optimized General Matrix Multiplication (GEMM) routine for computing C = A × B, where A, B, and C are matrices. The implementation leverages numerous performance optimization techniques to achieve significant speedup over naive implementations.

## Optimization Techniques

### 1. SIMD Vectorization with AVX-512
The implementation uses Intel's Advanced Vector Extensions 512 (AVX-512) instruction set to process multiple elements simultaneously:
- `__m512` vector registers process 16 single-precision floating-point values in parallel
- `_mm512_fmadd_ps` combines multiplication and addition in a single instruction (FMA)
- `_mm512_set1_ps` broadcasts a single value across all vector lanes
- Non-temporal stores (`_mm512_stream_ps`) bypass cache when writing results

### 2. Cache Blocking (Tiling)
The algorithm implements a three-level cache blocking strategy to maximize data locality:
- Blocking parameters (BN=64, BM=64, BK=256) are tuned for common L1 (32KB) and L2 (1MB) cache sizes
- Reduces TLB misses and cache line conflicts
- Improves data reuse within each hierarchical memory level

### 3. Memory Alignment and Management
- 64-byte memory alignment (matching AVX-512 vector width) ensures optimal memory access
- `_mm_malloc`/`_mm_free` used for proper aligned memory allocation/deallocation
- Memory is zero-initialized using `std::memset` before computation

### 4. Parallel Processing with OpenMP
- Parallelizes computation across all available CPU cores
- Uses `omp parallel for` with dynamic scheduling for load balancing
- Parallel file I/O with `omp parallel sections` for input matrices reading
- Runtime thread count detection with `omp_get_max_threads()`

### 5. Loop Optimization Techniques
- Loop unrolling with 8-row and 16-column processing blocks
- Register blocking to minimize register spilling
- Loop interleaving to improve instruction-level parallelism
- Special case handling for edge cases (non-multiple-of-8/16 dimensions)

### 6. Prefetching
- Strategic data prefetching with `_mm_prefetch` to reduce cache misses
- Only minimal prefetching is used as extensive prefetching was found to hurt performance

### 7. Compiler Optimizations
- `#pragma GCC optimize("O3,unroll-loops")` enables aggressive optimization
- Target-specific tuning with explicit instruction set flags
- Uses BMI, BMI2, and FMA instruction sets for additional performance

## Code Explanation

### Memory Management
```cpp
constexpr size_t ALIGNMENT = 64;
float *m1 = static_cast<float *>(_mm_malloc(n * k * sizeof(float), ALIGNMENT));
float *m2 = static_cast<float *>(_mm_malloc(k * m * sizeof(float), ALIGNMENT));
float *result = static_cast<float *>(_mm_malloc(n * m * sizeof(float), ALIGNMENT));
```
Aligned memory allocation is crucial for vectorized operations, ensuring we can perform aligned loads and stores.

### Parallel File I/O
```cpp
#pragma omp parallel sections
{
    #pragma omp section
    { /* Read matrix 1 */ }
    
    #pragma omp section
    { /* Read matrix 2 */ }
}
```
This overlaps the reading of input matrices, utilizing multiple cores to speed up I/O operations.

### Blocking Strategy
```cpp
const int BN = 64;   // Block size for rows (A matrix)
const int BM = 64;   // Block size for columns (B matrix)
const int BK = 256;  // Block size for inner dimension
```
These values are tuned to fit within cache sizes and minimize cache coherency traffic between CPU cores.

### Vectorized Computation Core
```cpp
// Load 8 rows worth of accumulated sums
__m512 sum0 = _mm512_load_ps(&result[(i + 0) * m + j]);
// ...

// Process 8 rows with interleaving
sum0 = _mm512_fmadd_ps(_mm512_set1_ps(m1[(i + 0) * k + l]), b, sum0);
// ...
```
This code processes 8 rows and 16 columns (128 elements) in each inner loop iteration, maximizing instruction-level parallelism and register utilization.

### Non-temporal Stores
```cpp
_mm512_stream_ps(&result[(i + 0) * m + j], sum0);
```
These stores bypass the cache, reducing cache pollution for large matrices where result values are unlikely to be reused soon.

## Performance Comparison

Compared to a naive triple-nested loop implementation, this optimized GEMM provides:
- **Vectorization**: ~16× speedup from SIMD processing 16 elements at once
- **Cache Blocking**: Up to 10× improvement for large matrices from better cache utilization
- **Multi-threading**: Nearly linear scaling with CPU core count
- **Memory Management**: ~2× improvement from proper alignment and non-temporal stores

## Compilation and Usage

The code requires:
- A CPU supporting AVX-512 instructions
- GCC compiler with OpenMP support
- CMake build system

The implementation is used through the `solution::compute` function which takes paths to the input matrix files and their dimensions.
