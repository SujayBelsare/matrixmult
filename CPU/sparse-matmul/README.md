# Sparse Matrix-Matrix Multiplication Optimization

This repository contains an optimized implementation of Sparse Matrix-Matrix Multiplication (SpMM) using the Compressed Sparse Row (CSR) format. The code implements several performance optimizations to accelerate sparse matrix operations.

## Algorithm Overview

The implementation uses the CSR format to efficiently represent and multiply sparse matrices. The CSR format stores only non-zero elements along with their positions, significantly reducing memory usage and computational requirements for sparse matrices.

### Key Components

1. **CSR Matrix Representation**: Stores matrices using three arrays:
   - `values`: Non-zero values
   - `column_indices`: Column indices for the non-zero values
   - `row_pointers`: Starting positions for each row in the sparse matrix

2. **Dense to CSR Conversion**: Transforms standard dense matrices into the CSR format by keeping only non-zero values and their positions.

3. **Sparse Matrix Multiplication**: Efficiently multiplies two sparse matrices in CSR format.

## Optimizations

### 1. Sparse Format (CSR)

- **What**: Represents matrices using the Compressed Sparse Row format
- **Why**: Drastically reduces memory usage and computation time for sparse matrices by only storing and processing non-zero elements
- **Impact**: Memory usage scales with the number of non-zeros rather than matrix dimensions

### 2. OpenMP Parallelization

- **What**: Multiple parallelization techniques using OpenMP
- **Why**: Utilizes multi-core processors efficiently
- **Impact**: Near-linear speedup with number of cores

Specific parallelization techniques:
- **Parallel For with Dynamic Scheduling**: 
  ```cpp
  #pragma omp parallel for schedule(dynamic, 16)
  ```
  Distributes SpMM computation across threads with dynamic load balancing. The chunk size of 16 is chosen to balance overhead with load distribution.

- **Parallel Sections**:
  ```cpp
  #pragma omp parallel sections
  ```
  Used for concurrent I/O operations and matrix conversions, allowing these operations to execute in parallel.

### 3. Memory Alignment

- **What**: 64-byte aligned memory allocation using `_mm_malloc`
- **Why**: Enables efficient SIMD (Single Instruction Multiple Data) operations
- **Impact**: Improves cache utilization and enables vectorization

```cpp
float *m1 = static_cast<float *>(_mm_malloc(n * k * sizeof(float), ALIGNMENT));
```

### 4. Compiler Optimizations

- **What**: GCC-specific optimization directives
- **Why**: Instructs the compiler to use specific optimizations and hardware features
- **Impact**: Generates highly optimized machine code

```cpp
#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
```

- **O3**: Aggressive optimization level
- **unroll-loops**: Unrolls loops to reduce branch prediction misses
- **avx2**: Uses Advanced Vector Extensions 2 for SIMD operations
- **bmi, bmi2, lzcnt, popcnt**: Uses specific CPU instructions for bit manipulation

### 5. Dynamic Scheduling

- **What**: OpenMP dynamic scheduling with chunk size of 16
- **Why**: Different rows in sparse matrices may have widely varying numbers of non-zeros
- **Impact**: Balances work across threads to prevent any single thread from becoming a bottleneck

## Code Explanation

### Matrix Storage and Conversion

```cpp
struct CSRMatrix {
    std::vector<float> values;       // Non-zero values
    std::vector<int> column_indices; // Column indices of non-zero values
    std::vector<int> row_pointers;   // Starting positions of rows
    int rows;
    int cols;
};
```

The CSR format uses three arrays to store a sparse matrix efficiently:
- `values`: Stores all non-zero values
- `column_indices`: Stores the column index of each non-zero value
- `row_pointers`: Stores the starting position of each row in the `values` array

### SpMM Implementation

The core SpMM algorithm iterates through each non-zero element in matrix A, finds the corresponding row in matrix B, and computes the contribution to the result matrix:

```cpp
void spmm_csr(const CSRMatrix &A, const CSRMatrix &B, float *result, int n, int m) {
    // Initialize result to zeros
    std::memset(result, 0, sizeof(float) * n * m);

    #pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < A.rows; ++i) {
        // For each non-zero element in row i of A
        for (int j = A.row_pointers[i]; j < A.row_pointers[i + 1]; ++j) {
            int k = A.column_indices[j]; // Column index in A, row index in B
            float val_A = A.values[j];

            // For each non-zero element in row k of B
            for (int l = B.row_pointers[k]; l < B.row_pointers[k + 1]; ++l) {
                int col_B = B.column_indices[l];
                float val_B = B.values[l];

                // Update result
                result[i * B.cols + col_B] += val_A * val_B;
            }
        }
    }
}
```

This implementation is highly efficient because:
1. It only processes non-zero elements
2. Each thread works on independent rows of the result matrix (no race conditions)
3. Dynamic scheduling handles workload imbalance
4. Memory access patterns are optimized for cache performance

## Performance Considerations

- **Sparsity**: Performance gains increase with matrix sparsity
- **Threading Overhead**: For very small matrices, parallelization overhead might outweigh benefits
- **Memory Bandwidth**: SpMM performance is often memory-bound; aligned memory helps improve bandwidth utilization
- **Load Balancing**: Dynamic scheduling is crucial for matrices with irregular sparsity patterns

## Usage

The provided implementation takes two matrix files (in dense binary format), dimensions (n, k, m), and returns the path to the output file:

```cpp
std::string solution::compute(const std::string &m1_path, const std::string &m2_path, int n, int k, int m)
```