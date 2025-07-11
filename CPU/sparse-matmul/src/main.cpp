// Enable aggressive compiler optimizations for maximum performance
#pragma GCC optimize("O3,unroll-loops")
// Target modern CPU instruction sets for vectorization
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")

// Include headers for SIMD intrinsics, I/O, memory management, and parallel processing
#include <immintrin.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>
#include <cstring>
#include <omp.h> // OpenMP for parallel processing

// CSR (Compressed Sparse Row) Matrix representation
// This format stores only non-zero elements to save memory and computation
struct CSRMatrix
{
	std::vector<float> values;		 // Array of non-zero values in row-major order
	std::vector<int> column_indices; // Column indices corresponding to each non-zero value
	std::vector<int> row_pointers;	 // Starting index in values/column_indices for each row
	int rows;						 // Number of rows in the matrix
	int cols;						 // Number of columns in the matrix
};

// Convert a dense matrix (stored in row-major order) to CSR format
// This optimization reduces memory usage and computational complexity for sparse matrices
inline CSRMatrix dense_to_csr(const float *dense, int rows, int cols)
{
	CSRMatrix csr;
	csr.rows = rows;
	csr.cols = cols;

	// Initialize with first row starting at index 0
	csr.row_pointers.push_back(0);

	// Iterate through each element of the dense matrix
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			float val = dense[i * cols + j]; // Access element at (i,j) in row-major order
			if (val != 0) // Only store non-zero elements
			{
				csr.values.push_back(val);		  // Store the non-zero value
				csr.column_indices.push_back(j); // Store its column index
			}
		}
		// Record the starting position of the next row
		csr.row_pointers.push_back(csr.values.size());
	}

	return csr;
}

// Sparse matrix-matrix multiplication using CSR format with OpenMP parallelization
// Implements the algorithm: C = A * B where A and B are sparse matrices in CSR format
// Time complexity: O(nnz(A) * avg_nnz_per_row(B)) instead of O(n^3) for dense multiplication
inline void spmm_csr(const CSRMatrix &A, const CSRMatrix &B, float *result, int n, int m)
{
	// Initialize result matrix to zeros (required for accumulation)
	std::memset(result, 0, sizeof(float) * n * m);

	// Parallelize over rows of matrix A using dynamic scheduling
	// Dynamic scheduling balances load when rows have varying sparsity
#pragma omp parallel for schedule(dynamic, 16)
	for (int i = 0; i < A.rows; ++i)
	{
		// Process each non-zero element in row i of matrix A
		for (int j = A.row_pointers[i]; j < A.row_pointers[i + 1]; ++j)
		{
			int k = A.column_indices[j]; // Column index in A corresponds to row index in B
			float val_A = A.values[j];   // Non-zero value from matrix A

			// Multiply A[i,k] with each non-zero element in row k of matrix B
			for (int l = B.row_pointers[k]; l < B.row_pointers[k + 1]; ++l)
			{
				int col_B = B.column_indices[l]; // Column index in matrix B
				float val_B = B.values[l];		 // Non-zero value from matrix B

				// Accumulate the product: result[i,col_B] += A[i,k] * B[k,col_B]
				// Thread-safe since each thread processes different rows of the result
				result[i * B.cols + col_B] += val_A * val_B;
			}
		}
	}
}

namespace solution
{
	// Main computation function that performs sparse matrix multiplication
	// Parameters: m1_path, m2_path - binary files containing input matrices
	//            n, k, m - matrix dimensions (n×k) * (k×m) = (n×m)
	// Returns: path to the result file containing the output matrix
	std::string compute(const std::string &m1_path, const std::string &m2_path, int n, int k, int m)
	{
		// Create temporary file path for storing the result matrix
		std::string sol_path = (std::filesystem::temp_directory_path() / "student_sol.dat").string();
		std::ofstream sol_fs(sol_path, std::ios::binary);

		// Memory alignment for optimal SIMD performance (64-byte for AVX-512)
		constexpr size_t ALIGNMENT = 64;

		// Allocate aligned memory for matrices to enable vectorization
		float *m1 = static_cast<float *>(_mm_malloc(n * k * sizeof(float), ALIGNMENT));
		float *m2 = static_cast<float *>(_mm_malloc(k * m * sizeof(float), ALIGNMENT));
		float *result = static_cast<float *>(_mm_malloc(n * m * sizeof(float), ALIGNMENT));

		// Initialize result matrix to zeros before accumulation
		std::memset(result, 0, sizeof(float) * n * m);

		// Parallel I/O: read both input matrices simultaneously using OpenMP sections
#pragma omp parallel sections
		{
#pragma omp section
			{
				// Read first matrix from binary file
				std::ifstream m1_fs(m1_path, std::ios::binary);
				m1_fs.read(reinterpret_cast<char *>(m1), sizeof(float) * n * k);
				m1_fs.close();
			}

#pragma omp section
			{
				// Read second matrix from binary file
				std::ifstream m2_fs(m2_path, std::ios::binary);
				m2_fs.read(reinterpret_cast<char *>(m2), sizeof(float) * k * m);
				m2_fs.close();
			}
		}

		// Declare CSR matrices for sparse representation
		CSRMatrix A;
		CSRMatrix B;

		// Parallel conversion: convert both dense matrices to CSR format simultaneously
#pragma omp parallel sections
		{
#pragma omp section
			{
				A = dense_to_csr(m1, n, k); // Convert first matrix to CSR
			}
#pragma omp section
			{
				B = dense_to_csr(m2, k, m); // Convert second matrix to CSR
			}
		}

		// Perform optimized sparse matrix multiplication
		spmm_csr(A, B, result, n, m);

		// Write the result matrix to binary file
		sol_fs.write(reinterpret_cast<const char *>(result), sizeof(float) * n * m);
		sol_fs.close();

		// Clean up allocated memory to prevent memory leaks
		_mm_free(m1);
		_mm_free(m2);
		_mm_free(result);

		return sol_path;
	}
}