#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx512f,avx512dq,avx512bw,avx512vl,avx512cd,avx512vnni,bmi,bmi2,fma")

#include <immintrin.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <filesystem>
#include <string>
#include <omp.h>
#include <cstring>

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

		// Cache blocking parameters optimized for your specific cache sizes
		// Optimized for 32KB L1D and 1MB L2 cache per core
		const int BN = 64;	// Block size for rows (A matrix)
		const int BM = 64;	// Block size for columns (B matrix)
		const int BK = 256; // Block size for inner dimension

		// Use all available cores (detected at runtime)
		int num_threads = omp_get_max_threads();
		omp_set_num_threads(num_threads);

#pragma omp parallel
		{
#pragma omp for schedule(dynamic, 1)
			for (int i0 = 0; i0 < n; i0 += BN)
			{
				for (int j0 = 0; j0 < m; j0 += BM)
				{
					for (int k0 = 0; k0 < k; k0 += BK)
					{
						int i_end = std::min(i0 + BN, n);
						int j_end = std::min(j0 + BM, m);
						int k_end = std::min(k0 + BK, k);

						// 8-row interleaving for better register utilization
						for (int i = i0; i < i_end; i += 8)
						{
							if (i + 8 <= i_end) // Full 8-row block
							{
								for (int j = j0; j < j_end; j += 16)
								{
									if (j + 16 <= j_end) // Full 16-column block
									{
										// Load 8 rows worth of accumulated sums
										__m512 sum0 = _mm512_load_ps(&result[(i + 0) * m + j]);
										__m512 sum1 = _mm512_load_ps(&result[(i + 1) * m + j]);
										__m512 sum2 = _mm512_load_ps(&result[(i + 2) * m + j]);
										__m512 sum3 = _mm512_load_ps(&result[(i + 3) * m + j]);
										__m512 sum4 = _mm512_load_ps(&result[(i + 4) * m + j]);
										__m512 sum5 = _mm512_load_ps(&result[(i + 5) * m + j]);
										__m512 sum6 = _mm512_load_ps(&result[(i + 6) * m + j]);
										__m512 sum7 = _mm512_load_ps(&result[(i + 7) * m + j]);

										for (int l = k0; l < k_end; ++l)
										{
											// Minimal prefetch - since extensive prefetching hurt performance
											if (l + 8 < k_end)
											{
												_mm_prefetch(&m2[(l + 8) * m + j], _MM_HINT_T0);
											}

											// Load B matrix values (reused for all 8 rows)
											__m512 b = _mm512_load_ps(&m2[l * m + j]);

											// Process 8 rows with interleaving (high instruction-level parallelism)
											sum0 = _mm512_fmadd_ps(_mm512_set1_ps(m1[(i + 0) * k + l]), b, sum0);
											sum1 = _mm512_fmadd_ps(_mm512_set1_ps(m1[(i + 1) * k + l]), b, sum1);
											sum2 = _mm512_fmadd_ps(_mm512_set1_ps(m1[(i + 2) * k + l]), b, sum2);
											sum3 = _mm512_fmadd_ps(_mm512_set1_ps(m1[(i + 3) * k + l]), b, sum3);
											sum4 = _mm512_fmadd_ps(_mm512_set1_ps(m1[(i + 4) * k + l]), b, sum4);
											sum5 = _mm512_fmadd_ps(_mm512_set1_ps(m1[(i + 5) * k + l]), b, sum5);
											sum6 = _mm512_fmadd_ps(_mm512_set1_ps(m1[(i + 6) * k + l]), b, sum6);
											sum7 = _mm512_fmadd_ps(_mm512_set1_ps(m1[(i + 7) * k + l]), b, sum7);
										}

										// Use non-temporal stores for result matrix to bypass cache
										// since these values won't be reused soon
										_mm512_stream_ps(&result[(i + 0) * m + j], sum0);
										_mm512_stream_ps(&result[(i + 1) * m + j], sum1);
										_mm512_stream_ps(&result[(i + 2) * m + j], sum2);
										_mm512_stream_ps(&result[(i + 3) * m + j], sum3);
										_mm512_stream_ps(&result[(i + 4) * m + j], sum4);
										_mm512_stream_ps(&result[(i + 5) * m + j], sum5);
										_mm512_stream_ps(&result[(i + 6) * m + j], sum6);
										_mm512_stream_ps(&result[(i + 7) * m + j], sum7);
									}
									else // Handle remaining columns (< 16)
									{
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
							else // Handle remaining rows (< 8)
							{
								for (int ii = i; ii < i_end; ++ii)
								{
									for (int j = j0; j < j_end; j += 16)
									{
										if (j + 16 <= j_end)
										{
											__m512 sum = _mm512_load_ps(&result[ii * m + j]);

											for (int l = k0; l < k_end; ++l)
											{
												__m512 a = _mm512_set1_ps(m1[ii * k + l]);
												__m512 b = _mm512_load_ps(&m2[l * m + j]);
												sum = _mm512_fmadd_ps(a, b, sum);
											}

											_mm512_stream_ps(&result[ii * m + j], sum);
										}
										else
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
						}
					}
				}
			}
		}

		// Ensure all stores are visible
		_mm_sfence();

		sol_fs.write(reinterpret_cast<const char *>(result), sizeof(float) * n * m);
		sol_fs.close();

		// Free aligned memory
		_mm_free(m1);
		_mm_free(m2);
		_mm_free(result);

		return sol_path;
	}
};