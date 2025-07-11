#include <iostream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <filesystem>
#include <string>

// CUDA runtime API for GPU memory management and kernel execution
#include <cuda_runtime.h>
// Cooperative groups for advanced thread synchronization and warp-level operations
#include <cooperative_groups.h>

// Optimized block size - increased for better occupancy
// 256 threads per block provides good balance between occupancy and resource usage
#define BLOCK_DIM 256

// Larger batch size for better amortization of launch overhead
// Processing data in larger chunks reduces the overhead of multiple kernel launches
#define BATCH_SIZE 64

// Improved kernel using warp-level operations and better memory access patterns
__global__ void computeHistogramKernel(const int *__restrict__ input,
                                       unsigned int *__restrict__ histogram,
                                       int N, int B, int offset, int elements_to_process)
{
    // Use cooperative groups for explicit thread synchronization
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();

    // Calculate global thread ID for this thread across all blocks
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Grid stride pattern allows handling arrays larger than the grid size
    int stride = blockDim.x * gridDim.x;
    
    // Extract warp-level information for potential future optimizations
    const unsigned int lane_id = threadIdx.x % 32;  // Thread ID within warp (0-31)
    const unsigned int warp_id = threadIdx.x / 32;  // Warp ID within block

    // Shared memory for local histogram - each block maintains its own histogram
    // This reduces contention on global memory by aggregating locally first
    extern __shared__ unsigned int s_hist[];

// Initialize shared memory histogram with more efficient loop
// Use unroll pragma to reduce loop overhead and improve performance
// All threads in the block cooperatively initialize the shared histogram to zero
#pragma unroll 4
    for (int i = threadIdx.x; i < B; i += blockDim.x)
    {
        s_hist[i] = 0;
    }

    // Synchronize all threads in the block before proceeding to data processing
    // Ensures shared memory initialization is complete
    block.sync();

    // Process input data with coalesced memory access pattern
    // Grid-stride loop allows processing arrays larger than the total number of threads
    // Each thread processes multiple elements spaced by the grid stride
    for (int i = tid; i < elements_to_process; i += stride)
    {
        // Load input value - memory accesses are coalesced for better bandwidth utilization
        int value = input[i + offset];

        // Bounds checking to ensure histogram array access is safe
        // Only valid histogram bin indices are processed
        if (value >= 0 && value < B)
        {
            // Atomic increment to shared memory histogram
            // Multiple threads may increment the same bin, so atomics ensure correctness
            atomicAdd(&s_hist[value], 1u);
        }
    }

    // Synchronize all threads before merging local histograms
    // Ensures all threads have finished processing their assigned data
    block.sync();

    // Merge local histogram into global histogram with warp-level aggregation
    // to reduce atomic contention on global memory
    // Each thread handles a subset of histogram bins
    for (int i = threadIdx.x; i < B; i += blockDim.x)
    {
        // Read the accumulated count from shared memory
        unsigned int val = s_hist[i];
        
        // Only perform expensive global atomic operation if there's actually a count to add
        // This optimization reduces unnecessary atomic operations
        if (val > 0)
        {
            // Atomic add to global histogram - this is where results from all blocks are combined
            atomicAdd(&histogram[i], val);
        }
    }
}

namespace solution
{
    // Main histogram computation function
    std::string compute(const std::string &input_path, int N, int B)
    {
        // Generate output file path in system temporary directory
        std::string sol_path = std::filesystem::temp_directory_path() / "student_histogram.dat";

        // GPU initialization and device property querying
        cudaSetDevice(0);  // Use first available GPU
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);

        // Calculate optimal batching parameters based on available memory and device capabilities
        // Process data in chunks to handle datasets larger than GPU memory
        const int elements_per_batch = BATCH_SIZE * BLOCK_DIM * 1024;
        const int num_batches = (N + elements_per_batch - 1) / elements_per_batch;

        // Create multiple CUDA streams for overlapping operations
        // Stream 0: Memory operations, Stream 1&2: Computation
        cudaStream_t streams[3];
        for (int i = 0; i < 3; i++)
        {
            cudaStreamCreate(&streams[i]);
        }

        // Configure kernel execution parameters - dynamically adjust based on device
        const int threadsPerBlock = BLOCK_DIM;
        // Use more blocks to increase parallelism, but limit to max supported
        // Optimal block count balances occupancy with resource utilization
        const int blocks = std::min(deviceProp.multiProcessorCount * 32,
                                    (elements_per_batch + threadsPerBlock - 1) / threadsPerBlock);

        // Allocate device memory for histogram and zero it
        // Global histogram accumulates results from all blocks and batches
        unsigned int *d_histogram;
        cudaMalloc(&d_histogram, sizeof(unsigned int) * B);
        cudaMemsetAsync(d_histogram, 0, sizeof(unsigned int) * B, streams[0]);

        // Use pinned (page-locked) memory for host buffers to enable faster transfer
        // Pinned memory provides higher bandwidth for GPU-CPU transfers
        int *h_input_buffers[2];
        cudaMallocHost(&h_input_buffers[0], sizeof(int) * elements_per_batch);
        cudaMallocHost(&h_input_buffers[1], sizeof(int) * elements_per_batch);

        // Allocate device memory for input - double buffering
        // Double buffering allows overlapping data transfer with computation
        int *d_input_buffers[2];
        cudaMalloc(&d_input_buffers[0], sizeof(int) * elements_per_batch);
        cudaMalloc(&d_input_buffers[1], sizeof(int) * elements_per_batch);

        // Final histogram will be stored here - use pinned memory for fast transfer
        unsigned int *h_histogram;
        cudaMallocHost(&h_histogram, sizeof(unsigned int) * B);

        // Open input file in binary mode for reading integer data
        std::ifstream input_fs(input_path, std::ios::binary);
        if (!input_fs)
        {
            std::cerr << "Error: Could not open input file " << input_path << std::endl;
            // Cleanup resources before returning on error
            // Proper resource management prevents memory leaks
            cudaFreeHost(h_input_buffers[0]);
            cudaFreeHost(h_input_buffers[1]);
            cudaFreeHost(h_histogram);
            cudaFree(d_input_buffers[0]);
            cudaFree(d_input_buffers[1]);
            cudaFree(d_histogram);
            for (int i = 0; i < 3; i++)
            {
                cudaStreamDestroy(streams[i]);
            }
            return "";
        }

        // Process data in batches with triple buffering strategy
        // Create events for fine-grained synchronization between operations
        // Events allow tracking completion of specific operations across streams
        cudaEvent_t events[num_batches];
        for (int i = 0; i < num_batches; i++)
        {
            cudaEventCreate(&events[i]);
        }

        // Main processing loop implementing pipelined execution
        // This loop overlaps I/O, memory transfer, and computation for maximum throughput
        int prev_buffer_idx = -1;
        for (int batch = 0; batch < num_batches; batch++)
        {
            // Alternate between two buffers for double buffering
            int buffer_idx = batch % 2;
            int offset = batch * elements_per_batch;
            // Handle the last batch which may have fewer elements
            int elements_this_batch = std::min(elements_per_batch, N - offset);
            size_t bytes_to_read = elements_this_batch * sizeof(int);

            // Read batch of input data from file into host memory
            // File I/O can be overlapped with GPU computation of previous batch
            input_fs.read(reinterpret_cast<char *>(h_input_buffers[buffer_idx]), bytes_to_read);

            // Use round-robin assignment of streams for load balancing
            int stream_idx = batch % 3;

            // Transfer current batch from host to device memory asynchronously
            // Asynchronous transfer allows overlapping with other operations
            cudaMemcpyAsync(d_input_buffers[buffer_idx], h_input_buffers[buffer_idx],
                            bytes_to_read, cudaMemcpyHostToDevice, streams[stream_idx]);

            // Process previous batch while current batch is being transferred
            // This overlap of computation and I/O maximizes GPU utilization
            if (prev_buffer_idx >= 0)
            {
                // Calculate parameters for the previous batch
                int prev_offset = (batch - 1) * elements_per_batch;
                int prev_elements = std::min(elements_per_batch, N - prev_offset);
                int prev_stream_idx = (batch - 1) % 3;

                // Calculate shared memory size needed for histogram bins
                // Each block needs space for B unsigned integers
                size_t shared_mem_size = sizeof(unsigned int) * B;

                // Launch histogram computation kernel for previous batch
                // Kernel processes previous batch while current batch transfers
                computeHistogramKernel<<<blocks, threadsPerBlock, shared_mem_size, streams[prev_stream_idx]>>>(
                    d_input_buffers[prev_buffer_idx], d_histogram, N, B, 0, prev_elements);

                // Record completion event for this batch for later synchronization
                cudaEventRecord(events[batch - 1], streams[prev_stream_idx]);
            }

            // Update buffer index for next iteration
            prev_buffer_idx = buffer_idx;
        }

        // Process the last batch separately since it's not handled in the main loop
        // The last batch needs special handling because there's no "next" batch to overlap with
        if (num_batches > 0)
        {
            int last_buffer_idx = (num_batches - 1) % 2;
            int last_offset = (num_batches - 1) * elements_per_batch;
            int last_elements = std::min(elements_per_batch, N - last_offset);
            int last_stream_idx = (num_batches - 1) % 3;

            // Launch kernel for the final batch
            size_t shared_mem_size = sizeof(unsigned int) * B;
            computeHistogramKernel<<<blocks, threadsPerBlock, shared_mem_size, streams[last_stream_idx]>>>(
                d_input_buffers[last_buffer_idx], d_histogram, N, B, 0, last_elements);

            // Record completion event for the last batch
            cudaEventRecord(events[num_batches - 1], streams[last_stream_idx]);
        }

        // Wait for all computations to complete by waiting for last event
        // This ensures all histogram computations are finished before reading results
        if (num_batches > 0)
        {
            cudaEventSynchronize(events[num_batches - 1]);
        }

        // Copy final histogram from device to host using stream 0
        // Transfer the accumulated histogram results back to CPU memory
        cudaMemcpyAsync(h_histogram, d_histogram, sizeof(unsigned int) * B,
                        cudaMemcpyDeviceToHost, streams[0]);
        cudaStreamSynchronize(streams[0]);

        // Write output - convert to int for file format compatibility
        // Open output file in binary mode for writing histogram data
        std::ofstream sol_fs(sol_path, std::ios::binary);
        if (!sol_fs)
        {
            std::cerr << "Error: Could not open output file " << sol_path << std::endl;
            // Cleanup resources before returning on error
            cudaFreeHost(h_input_buffers[0]);
            cudaFreeHost(h_input_buffers[1]);
            cudaFreeHost(h_histogram);
            cudaFree(d_input_buffers[0]);
            cudaFree(d_input_buffers[1]);
            cudaFree(d_histogram);
            for (int i = 0; i < 3; i++)
            {
                cudaStreamDestroy(streams[i]);
            }
            for (int i = 0; i < num_batches; i++)
            {
                cudaEventDestroy(events[i]);
            }
            return "";
        }

        // Convert unsigned int histogram to signed int for output format compatibility
        // Some systems may expect signed integer output format
        auto int_histogram = std::make_unique<int[]>(B);
        for (int i = 0; i < B; i++)
        {
            int_histogram[i] = static_cast<int>(h_histogram[i]);
        }
        // Write the histogram data to output file in binary format
        sol_fs.write(reinterpret_cast<const char *>(int_histogram.get()), sizeof(int) * B);
        sol_fs.close();

        // Proper cleanup of all allocated resources
        // Free pinned host memory allocated with cudaMallocHost
        cudaFreeHost(h_input_buffers[0]);
        cudaFreeHost(h_input_buffers[1]);
        cudaFreeHost(h_histogram);
        
        // Free device memory allocated with cudaMalloc
        cudaFree(d_input_buffers[0]);
        cudaFree(d_input_buffers[1]);
        cudaFree(d_histogram);
        
        // Destroy CUDA streams to free associated resources
        for (int i = 0; i < 3; i++)
        {
            cudaStreamDestroy(streams[i]);
        }
        
        // Destroy CUDA events to free associated resources
        for (int i = 0; i < num_batches; i++)
        {
            cudaEventDestroy(events[i]);
        }

        // Return path to the generated histogram file
        return sol_path;
    }
}