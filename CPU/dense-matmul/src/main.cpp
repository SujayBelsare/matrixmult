// Compiler optimization pragmas for maximum performance
#pragma GCC optimize("O3,unroll-loops")  // Enable aggressive optimizations and loop unrolling
#pragma GCC target("avx512f,avx512dq,avx512bw,avx512vl,avx512cd,avx512vnni,bmi,bmi2,fma")  // Target modern CPU instruction sets

#include <immintrin.h>  // Intel intrinsics for SIMD operations
#include <iostream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <filesystem>
#include <string>
#include <omp.h>       // OpenMP for parallel processing
#include <cstring>     // For memset

namespace solution
{
	// Main computation function: performs optimized dense matrix multiplication
	// Multiplies matrix m1 (n×k) with matrix m2 (k×m) to produce result (n×m)
	std::string compute(const std::string &m1_path, const std::string &m2_path, int n, int k, int m)
	{
		// Create output file path for the result matrix
		std::string sol_path = (std::filesystem::temp_directory_path() / "student_sol.dat").string();
		std::ofstream sol_fs(sol_path, std::ios::binary);

		// Allocate aligned memory (64-byte alignment for AVX-512 optimal performance)
		// Aligned memory ensures SIMD loads/stores are not split across cache lines
		constexpr size_t ALIGNMENT = 64;

		float *m1 = static_cast<float *>(_mm_malloc(n * k * sizeof(float), ALIGNMENT));
		float *m2 = static_cast<float *>(_mm_malloc(k * m * sizeof(float), ALIGNMENT));
		float *result = static_cast<float *>(_mm_malloc(n * m * sizeof(float), ALIGNMENT));

		// Initialize result matrix to zeros (required for accumulation)
		std::memset(result, 0, sizeof(float) * n * m);

// Read input matrices in parallel to overlap I/O operations
// Using OpenMP sections to read both matrices simultaneously
#pragma omp parallel sections
		{
#pragma omp section
			{
				// Read first matrix (m1) from binary file
				std::ifstream m1_fs(m1_path, std::ios::binary);
				m1_fs.read(reinterpret_cast<char *>(m1), sizeof(float) * n * k);
				m1_fs.close();
			}

#pragma omp section
			{
				// Read second matrix (m2) from binary file
				std::ifstream m2_fs(m2_path, std::ios::binary);
				m2_fs.read(reinterpret_cast<char *>(m2), sizeof(float) * k * m);
				m2_fs.close();
			}
		}

		// Cache blocking parameters optimized for typical CPU cache hierarchy
		// These values are tuned for 32KB L1D cache and 1MB L2 cache per core
		// Cache blocking improves data locality and reduces cache misses
		const int BN = 64;	// Block size for rows (A matrix) - fits in L1 cache
		const int BM = 64;	// Block size for columns (B matrix) - fits in L1 cache
		const int BK = 256; // Block size for inner dimension - balances reuse and cache pressure

		// Use all available CPU cores for maximum parallelism
		int num_threads = omp_get_max_threads();
		omp_set_num_threads(num_threads);

// Main parallel computation using OpenMP
#pragma omp parallel
		{
			// Dynamic scheduling distributes work evenly across threads
			// Each thread gets one block at a time to balance load
#pragma omp for schedule(dynamic, 1)
			for (int i0 = 0; i0 < n; i0 += BN)  // Iterate over row blocks
			{
				for (int j0 = 0; j0 < m; j0 += BM)  // Iterate over column blocks
				{
					for (int k0 = 0; k0 < k; k0 += BK)  // Iterate over inner dimension blocks
					{
						// Calculate actual block boundaries (handle edge cases)
						int i_end = std::min(i0 + BN, n);
						int j_end = std::min(j0 + BM, m);
						int k_end = std::min(k0 + BK, k);

						// 8-row interleaving optimization for better register utilization
						// Processing 8 rows simultaneously maximizes SIMD efficiency
						for (int i = i0; i < i_end; i += 8)
						{
							if (i + 8 <= i_end) // Full 8-row block processing
							{
								// Process columns in chunks of 16 (one AVX-512 vector)
								for (int j = j0; j < j_end; j += 16)
								{
									if (j + 16 <= j_end) // Full 16-column block processing
									{
										// Load 8 rows worth of accumulated sums from result matrix
										// Each __m512 register holds 16 float values (512 bits / 32 bits per float)
										__m512 sum0 = _mm512_load_ps(&result[(i + 0) * m + j]);
										__m512 sum1 = _mm512_load_ps(&result[(i + 1) * m + j]);
										__m512 sum2 = _mm512_load_ps(&result[(i + 2) * m + j]);
										__m512 sum3 = _mm512_load_ps(&result[(i + 3) * m + j]);
										__m512 sum4 = _mm512_load_ps(&result[(i + 4) * m + j]);
										__m512 sum5 = _mm512_load_ps(&result[(i + 5) * m + j]);
										__m512 sum6 = _mm512_load_ps(&result[(i + 6) * m + j]);
										__m512 sum7 = _mm512_load_ps(&result[(i + 7) * m + j]);

										// Inner loop: compute dot products for current block
										for (int l = k0; l < k_end; ++l)
										{
											// Conservative prefetching to improve memory access patterns
											// Prefetch next iteration's data into L1 cache
											if (l + 8 < k_end)
											{
												_mm_prefetch(&m2[(l + 8) * m + j], _MM_HINT_T0);
											}

											// Load 16 consecutive values from matrix B (reused across all 8 rows)
											__m512 b = _mm512_load_ps(&m2[l * m + j]);

											// Fused multiply-add operations for all 8 rows simultaneously
											// Each operation computes: sum = (a * b) + sum
											// _mm512_set1_ps broadcasts single value to all 16 lanes
											sum0 = _mm512_fmadd_ps(_mm512_set1_ps(m1[(i + 0) * k + l]), b, sum0);
											sum1 = _mm512_fmadd_ps(_mm512_set1_ps(m1[(i + 1) * k + l]), b, sum1);
											sum2 = _mm512_fmadd_ps(_mm512_set1_ps(m1[(i + 2) * k + l]), b, sum2);
											sum3 = _mm512_fmadd_ps(_mm512_set1_ps(m1[(i + 3) * k + l]), b, sum3);
											sum4 = _mm512_fmadd_ps(_mm512_set1_ps(m1[(i + 4) * k + l]), b, sum4);
											sum5 = _mm512_fmadd_ps(_mm512_set1_ps(m1[(i + 5) * k + l]), b, sum5);
											sum6 = _mm512_fmadd_ps(_mm512_set1_ps(m1[(i + 6) * k + l]), b, sum6);
											sum7 = _mm512_fmadd_ps(_mm512_set1_ps(m1[(i + 7) * k + l]), b, sum7);
										}

										// Store results using non-temporal stores to bypass cache
										// Non-temporal stores avoid cache pollution since result values
										// won't be accessed again soon (write-only pattern)
										_mm512_stream_ps(&result[(i + 0) * m + j], sum0);
										_mm512_stream_ps(&result[(i + 1) * m + j], sum1);
										_mm512_stream_ps(&result[(i + 2) * m + j], sum2);
										_mm512_stream_ps(&result[(i + 3) * m + j], sum3);
										_mm512_stream_ps(&result[(i + 4) * m + j], sum4);
										_mm512_stream_ps(&result[(i + 5) * m + j], sum5);
										_mm512_stream_ps(&result[(i + 6) * m + j], sum6);
										_mm512_stream_ps(&result[(i + 7) * m + j], sum7);
									}
									else // Handle remaining columns when j+16 > j_end (edge case)
									{
										// Fall back to scalar computation for irregular column sizes
										for (int ii = i; ii < i + 8; ++ii)
										{
											for (int jj = j; jj < j_end; ++jj)
											{
												float sum = result[ii * m + jj];
												for (int l = k0; l < k_end; ++l)
												{
													sum += m1[ii * k + l] * m2[l * m + jj];
												}
												result[ii * m + jj] = sum;
											}
										}
									}
								}
							}
							else // Handle remaining rows when i+8 > i_end (edge case)
							{
								// Process remaining rows (less than 8) individually
								for (int ii = i; ii < i_end; ++ii)
								{
									for (int j = j0; j < j_end; j += 16)
									{
										if (j + 16 <= j_end)  // Full 16-column SIMD processing
										{
											// Load accumulated sum for current row
											__m512 sum = _mm512_load_ps(&result[ii * m + j]);

											// Compute dot product using SIMD
											for (int l = k0; l < k_end; ++l)
											{
												// Broadcast single A matrix element to all 16 lanes
												__m512 a = _mm512_set1_ps(m1[ii * k + l]);
												// Load 16 B matrix elements
												__m512 b = _mm512_load_ps(&m2[l * m + j]);
												// Fused multiply-add: sum += a * b
												sum = _mm512_fmadd_ps(a, b, sum);
											}

											// Store result using non-temporal store
											_mm512_stream_ps(&result[ii * m + j], sum);
										}
										else  // Handle remaining columns for this row
										{
											// Fall back to scalar computation
											for (int jj = j; jj < j_end; ++jj)
											{
												float sum = result[ii * m + jj];
												for (int l = k0; l < k_end; ++l)
												{
													sum += m1[ii * k + l] * m2[l * m + jj];
												}
												result[ii * m + jj] = sum;
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}

		// Ensure all non-temporal stores are completed before proceeding
		// This memory fence guarantees all streaming stores reach memory
		_mm_sfence();

		// Write the computed result matrix to output file
		sol_fs.write(reinterpret_cast<const char *>(result), sizeof(float) * n * m);
		sol_fs.close();

		// Free all dynamically allocated aligned memory
		_mm_free(m1);
		_mm_free(m2);
		_mm_free(result);

		return sol_path;  // Return path to result file
	}
};