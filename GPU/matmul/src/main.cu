#include <iostream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <filesystem>
#include <string>
#include <cuda_runtime.h>

/**
 * Optimized CUDA kernel for matrix multiplication
 *
 * @param matrixA First input matrix (M x K)
 * @param matrixB Second input matrix (K x N)
 * @param resultMatrix Output matrix (M x N)
 * @param numRowsA Number of rows in matrix A
 * @param numColsB Number of columns in matrix B
 * @param numColsA_RowsB Number of columns in A / rows in B (shared dimension)
 */
__global__ void matrixMultiplyOptimized(
    const float *__restrict__ matrixA,
    const float *__restrict__ matrixB,
    float *__restrict__ resultMatrix,
    int numRowsA,
    int numColsB,
    int numColsA_RowsB)
{
    // Configuration constants
    const int TILE_SIZE = 64;          // Size of shared memory tile
    const int ELEMENTS_PER_THREAD = 4; // Each thread computes 4x4 elements

    // Shared memory tiles with padding to avoid bank conflicts
    __shared__ float sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedB[TILE_SIZE][TILE_SIZE + 1];

    // Calculate global indices for this thread's 4x4 block
    int globalRow = blockIdx.y * TILE_SIZE + threadIdx.y * ELEMENTS_PER_THREAD;
    int globalCol = blockIdx.x * TILE_SIZE + threadIdx.x * ELEMENTS_PER_THREAD;

    // Local accumulator for the 4x4 result elements
    float localResult[ELEMENTS_PER_THREAD][ELEMENTS_PER_THREAD] = {{0}};

    // Process tiles across the shared dimension
#pragma unroll 4
    for (int tileIdx = 0; tileIdx < (numColsA_RowsB + TILE_SIZE - 1) / TILE_SIZE; ++tileIdx)
    {
        // Load tiles into shared memory using float4 for coalesced access
#pragma unroll 4
        for (int i = 0; i < ELEMENTS_PER_THREAD; i++)
        {
            int rowA = globalRow + i;
            int colA = tileIdx * TILE_SIZE + threadIdx.x * ELEMENTS_PER_THREAD;

            // Load 4 consecutive elements from matrix A using float4
            float4 loadedA = reinterpret_cast<const float4 *>(&matrixA[rowA * numColsA_RowsB + colA])[0];
            sharedA[threadIdx.y * ELEMENTS_PER_THREAD + i][threadIdx.x * ELEMENTS_PER_THREAD] = loadedA.x;
            sharedA[threadIdx.y * ELEMENTS_PER_THREAD + i][threadIdx.x * ELEMENTS_PER_THREAD + 1] = loadedA.y;
            sharedA[threadIdx.y * ELEMENTS_PER_THREAD + i][threadIdx.x * ELEMENTS_PER_THREAD + 2] = loadedA.z;
            sharedA[threadIdx.y * ELEMENTS_PER_THREAD + i][threadIdx.x * ELEMENTS_PER_THREAD + 3] = loadedA.w;

            int rowB = tileIdx * TILE_SIZE + threadIdx.y * ELEMENTS_PER_THREAD + i;
            int colB = globalCol;

            // Load 4 consecutive elements from matrix B using float4
            float4 loadedB = reinterpret_cast<const float4 *>(&matrixB[rowB * numColsB + colB])[0];
            sharedB[threadIdx.y * ELEMENTS_PER_THREAD + i][threadIdx.x * ELEMENTS_PER_THREAD] = loadedB.x;
            sharedB[threadIdx.y * ELEMENTS_PER_THREAD + i][threadIdx.x * ELEMENTS_PER_THREAD + 1] = loadedB.y;
            sharedB[threadIdx.y * ELEMENTS_PER_THREAD + i][threadIdx.x * ELEMENTS_PER_THREAD + 2] = loadedB.z;
            sharedB[threadIdx.y * ELEMENTS_PER_THREAD + i][threadIdx.x * ELEMENTS_PER_THREAD + 3] = loadedB.w;
        }

        // Ensure all threads have finished loading to shared memory
        __syncthreads();

        // Compute partial products using register accumulation
#pragma unroll 4
        for (int k = 0; k < TILE_SIZE; k++)
        {
            float elementsA[ELEMENTS_PER_THREAD];
            float elementsB[ELEMENTS_PER_THREAD];

            // Load elements into registers for faster access
            for (int i = 0; i < ELEMENTS_PER_THREAD; i++)
            {
                elementsA[i] = sharedA[threadIdx.y * ELEMENTS_PER_THREAD + i][k];
                elementsB[i] = sharedB[k][threadIdx.x * ELEMENTS_PER_THREAD + i];
            }

            // Compute 4x4 outer product
            for (int i = 0; i < ELEMENTS_PER_THREAD; i++)
            {
                for (int j = 0; j < ELEMENTS_PER_THREAD; j++)
                {
                    localResult[i][j] += elementsA[i] * elementsB[j];
                }
            }
        }

        // Wait for all threads before loading new tiles
        __syncthreads();
    }

    // Store the 4x4 results to global memory
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++)
    {
        for (int j = 0; j < ELEMENTS_PER_THREAD; j++)
        {
            int resultRow = globalRow + i;
            int resultCol = globalCol + j;

            // Only write results that are within bounds
            if (resultRow < numRowsA && resultCol < numColsB)
            {
                resultMatrix[resultRow * numColsB + resultCol] = localResult[i][j];
            }
        }
    }
}

namespace solution
{
    /**
     * Computes the matrix multiplication of matrices stored in files
     *
     * @param matrix1Path Path to the first matrix file
     * @param matrix2Path Path to the second matrix file
     * @param numRowsA Number of rows in first matrix
     * @param numColsA_RowsB Columns of first matrix / rows of second matrix
     * @param numColsB Number of columns in second matrix
     * @return Path to the file containing the result matrix
     */
    std::string compute(
        const std::string &matrix1Path,
        const std::string &matrix2Path,
        int numRowsA,
        int numColsA_RowsB,
        int numColsB)
    {
        // Create output file path in temp directory
        std::string resultPath = std::filesystem::temp_directory_path() / "solution.dat";
        std::ofstream resultFile(resultPath, std::ios::binary);

        // Allocate host memory for matrices
        auto hostA = std::make_unique<float[]>(numRowsA * numColsA_RowsB);
        auto hostB = std::make_unique<float[]>(numColsA_RowsB * numColsB);
        auto hostResult = std::make_unique<float[]>(numRowsA * numColsB);

        // Read input matrices from files
        std::ifstream(matrix1Path).read(reinterpret_cast<char *>(hostA.get()), sizeof(float) * numRowsA * numColsA_RowsB);
        std::ifstream(matrix2Path).read(reinterpret_cast<char *>(hostB.get()), sizeof(float) * numColsA_RowsB * numColsB);

        // Allocate device memory
        float *deviceA, *deviceB, *deviceResult;
        cudaMalloc(&deviceA, sizeof(float) * numRowsA * numColsA_RowsB);
        cudaMalloc(&deviceB, sizeof(float) * numColsA_RowsB * numColsB);
        cudaMalloc(&deviceResult, sizeof(float) * numRowsA * numColsB);

        // Copy data to device
        cudaMemcpy(deviceA, hostA.get(), sizeof(float) * numRowsA * numColsA_RowsB, cudaMemcpyHostToDevice);
        cudaMemcpy(deviceB, hostB.get(), sizeof(float) * numColsA_RowsB * numColsB, cudaMemcpyHostToDevice);

        // Configure kernel execution
        const int TILE_SIZE = 64;
        const int THREAD_BLOCK_SIZE = 16; // 16x16 threads per block

        dim3 blockDimensions(THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE);
        dim3 gridDimensions(
            (numColsB + TILE_SIZE - 1) / TILE_SIZE,
            (numRowsA + TILE_SIZE - 1) / TILE_SIZE);

        // Launch kernel
        matrixMultiplyOptimized<<<gridDimensions, blockDimensions>>>(
            deviceA, deviceB, deviceResult, numRowsA, numColsB, numColsA_RowsB);

        // Copy result back to host
        cudaMemcpy(hostResult.get(), deviceResult, sizeof(float) * numRowsA * numColsB, cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(deviceA);
        cudaFree(deviceB);
        cudaFree(deviceResult);

        // Write result to file
        resultFile.write(reinterpret_cast<const char *>(hostResult.get()),
                         sizeof(float) * numRowsA * numColsB);

        return resultPath;
    }
}