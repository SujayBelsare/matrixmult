#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")

#include <immintrin.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>
#include <cstring>
#include <omp.h> // Added OpenMP header

// CSR Matrix representation
struct CSRMatrix
{
	std::vector<float> values;		 // Non-zero values
	std::vector<int> column_indices; // Column indices of non-zero values
	std::vector<int> row_pointers;	 // Starting positions of rows
	int rows;
	int cols;
};

// Convert dense matrix to CSR format
inline CSRMatrix dense_to_csr(const float *dense, int rows, int cols)
{
	CSRMatrix csr;
	csr.rows = rows;
	csr.cols = cols;

	csr.row_pointers.push_back(0);

	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			float val = dense[i * cols + j];
			if (val != 0)
			{
				csr.values.push_back(val);
				csr.column_indices.push_back(j);
			}
		}
		csr.row_pointers.push_back(csr.values.size());
	}

	return csr;
}

// Sparse matrix-matrix multiplication using CSR format with OpenMP
inline void spmm_csr(const CSRMatrix &A, const CSRMatrix &B, float *result, int n, int m)
{
	// Initialize result to zeros
	std::memset(result, 0, sizeof(float) * n * m);

// For each row in A, parallelize with dynamic scheduling
#pragma omp parallel for schedule(dynamic, 16)
	for (int i = 0; i < A.rows; ++i)
	{
		// For each non-zero element in row i of A
		for (int j = A.row_pointers[i]; j < A.row_pointers[i + 1]; ++j)
		{
			int k = A.column_indices[j]; // Column index in A, row index in B
			float val_A = A.values[j];

			// For each non-zero element in row k of B
			for (int l = B.row_pointers[k]; l < B.row_pointers[k + 1]; ++l)
			{
				int col_B = B.column_indices[l];
				float val_B = B.values[l];

				// Update result - no race condition since each thread works on different rows
				result[i * B.cols + col_B] += val_A * val_B;
			}
		}
	}
}

namespace solution
{
	std::string compute(const std::string &m1_path, const std::string &m2_path, int n, int k, int m)
	{
		std::string sol_path = std::filesystem::temp_directory_path() / "student_sol.dat";
		std::ofstream sol_fs(sol_path, std::ios::binary);

		// Allocate aligned memory (64-byte alignment for AVX-512)
		constexpr size_t ALIGNMENT = 64;

		float *m1 = static_cast<float *>(_mm_malloc(n * k * sizeof(float), ALIGNMENT));
		float *m2 = static_cast<float *>(_mm_malloc(k * m * sizeof(float), ALIGNMENT));
		float *result = static_cast<float *>(_mm_malloc(n * m * sizeof(float), ALIGNMENT));

		// Initialize result to zeros
		std::memset(result, 0, sizeof(float) * n * m);

// Read input matrices in parallel
#pragma omp parallel sections
		{
#pragma omp section
			{
				std::ifstream m1_fs(m1_path, std::ios::binary);
				m1_fs.read(reinterpret_cast<char *>(m1), sizeof(float) * n * k);
				m1_fs.close();
			}

#pragma omp section
			{
				std::ifstream m2_fs(m2_path, std::ios::binary);
				m2_fs.read(reinterpret_cast<char *>(m2), sizeof(float) * k * m);
				m2_fs.close();
			}
		}

		CSRMatrix A;
		CSRMatrix B;

#pragma omp parallel sections
		{
#pragma omp section
			{
				A = dense_to_csr(m1, n, k);
			}
#pragma omp section
			{
				B = dense_to_csr(m2, k, m);
			}
		}

		// Perform sparse matrix multiplication
		spmm_csr(A, B, result, n, m);

		sol_fs.write(reinterpret_cast<const char *>(result), sizeof(float) * n * m);
		sol_fs.close();

		// Free allocated memory
		_mm_free(m1);
		_mm_free(m2);
		_mm_free(result);

		return sol_path;
	}
};