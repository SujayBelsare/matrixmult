#include <iostream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <filesystem>
#include <string>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

// Optimized block size - increased for better occupancy
#define BLOCK_DIM 256
// Larger batch size for better amortization of launch overhead
#define BATCH_SIZE 64

// Improved kernel using warp-level operations and better memory access patterns
__global__ void computeHistogramKernel(const int *__restrict__ input,
                                       unsigned int *__restrict__ histogram,
                                       int N, int B, int offset, int elements_to_process)
{
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();

    // Calculate global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_id = threadIdx.x / 32;

    // Shared memory for local histogram
    extern __shared__ unsigned int s_hist[];

// Initialize shared memory histogram with more efficient loop
#pragma unroll 4
    for (int i = threadIdx.x; i < B; i += blockDim.x)
    {
        s_hist[i] = 0;
    }

    block.sync();

    // Process input data with coalesced memory access pattern
    for (int i = tid; i < elements_to_process; i += stride)
    {
        int value = input[i + offset];

        // Ensure value is within bounds of the histogram
        if (value >= 0 && value < B)
        {
            atomicAdd(&s_hist[value], 1u);
        }
    }

    block.sync();

    // Merge local histogram into global histogram with warp-level aggregation
    // to reduce atomic contention
    for (int i = threadIdx.x; i < B; i += blockDim.x)
    {
        unsigned int val = s_hist[i];
        if (val > 0)
        {
            atomicAdd(&histogram[i], val);
        }
    }
}

namespace solution
{
    std::string compute(const std::string &input_path, int N, int B)
    {
        std::string sol_path = std::filesystem::temp_directory_path() / "student_histogram.dat";

        // Select GPU device and get device properties
        cudaSetDevice(0);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);

        // Calculate optimal parameters based on device properties
        const int elements_per_batch = BATCH_SIZE * BLOCK_DIM * 1024;
        const int num_batches = (N + elements_per_batch - 1) / elements_per_batch;

        // Use 3 streams for better overlapping of operations
        cudaStream_t streams[3];
        for (int i = 0; i < 3; i++)
        {
            cudaStreamCreate(&streams[i]);
        }

        // Configure kernel execution parameters - dynamically adjust based on device
        const int threadsPerBlock = BLOCK_DIM;
        // Use more blocks to increase parallelism, but limit to max supported
        const int blocks = std::min(deviceProp.multiProcessorCount * 32,
                                    (elements_per_batch + threadsPerBlock - 1) / threadsPerBlock);

        // Allocate device memory for histogram and zero it
        unsigned int *d_histogram;
        cudaMalloc(&d_histogram, sizeof(unsigned int) * B);
        cudaMemsetAsync(d_histogram, 0, sizeof(unsigned int) * B, streams[0]);

        // Use pinned memory for host buffers to enable faster transfer
        int *h_input_buffers[2];
        cudaMallocHost(&h_input_buffers[0], sizeof(int) * elements_per_batch);
        cudaMallocHost(&h_input_buffers[1], sizeof(int) * elements_per_batch);

        // Allocate device memory for input - double buffering
        int *d_input_buffers[2];
        cudaMalloc(&d_input_buffers[0], sizeof(int) * elements_per_batch);
        cudaMalloc(&d_input_buffers[1], sizeof(int) * elements_per_batch);

        // Final histogram will be stored here
        unsigned int *h_histogram;
        cudaMallocHost(&h_histogram, sizeof(unsigned int) * B);

        // Open input file
        std::ifstream input_fs(input_path, std::ios::binary);
        if (!input_fs)
        {
            std::cerr << "Error: Could not open input file " << input_path << std::endl;
            // Cleanup resources before returning
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
        // Start with first batch read & transfer
        cudaEvent_t events[num_batches];
        for (int i = 0; i < num_batches; i++)
        {
            cudaEventCreate(&events[i]);
        }

        int prev_buffer_idx = -1;
        for (int batch = 0; batch < num_batches; batch++)
        {
            int buffer_idx = batch % 2;
            int offset = batch * elements_per_batch;
            int elements_this_batch = std::min(elements_per_batch, N - offset);
            size_t bytes_to_read = elements_this_batch * sizeof(int);

            // Read batch of input data
            input_fs.read(reinterpret_cast<char *>(h_input_buffers[buffer_idx]), bytes_to_read);

            // Use events for better synchronization
            int stream_idx = batch % 3;

            // Transfer batch to device
            cudaMemcpyAsync(d_input_buffers[buffer_idx], h_input_buffers[buffer_idx],
                            bytes_to_read, cudaMemcpyHostToDevice, streams[stream_idx]);

            // Process previous batch while current batch is being transferred
            if (prev_buffer_idx >= 0)
            {
                // Previous batch should be in the device by now
                int prev_offset = (batch - 1) * elements_per_batch;
                int prev_elements = std::min(elements_per_batch, N - prev_offset);
                int prev_stream_idx = (batch - 1) % 3;

                // Use correct shared memory size based on data type
                size_t shared_mem_size = sizeof(unsigned int) * B;

                // Launch kernel with actual offset parameter
                computeHistogramKernel<<<blocks, threadsPerBlock, shared_mem_size, streams[prev_stream_idx]>>>(
                    d_input_buffers[prev_buffer_idx], d_histogram, N, B, 0, prev_elements);

                // Record event for this batch completion
                cudaEventRecord(events[batch - 1], streams[prev_stream_idx]);
            }

            prev_buffer_idx = buffer_idx;
        }

        // Process the last batch
        if (num_batches > 0)
        {
            int last_buffer_idx = (num_batches - 1) % 2;
            int last_offset = (num_batches - 1) * elements_per_batch;
            int last_elements = std::min(elements_per_batch, N - last_offset);
            int last_stream_idx = (num_batches - 1) % 3;

            size_t shared_mem_size = sizeof(unsigned int) * B;
            computeHistogramKernel<<<blocks, threadsPerBlock, shared_mem_size, streams[last_stream_idx]>>>(
                d_input_buffers[last_buffer_idx], d_histogram, N, B, 0, last_elements);

            cudaEventRecord(events[num_batches - 1], streams[last_stream_idx]);
        }

        // Wait for all computations to complete by waiting for last event
        if (num_batches > 0)
        {
            cudaEventSynchronize(events[num_batches - 1]);
        }

        // Copy final histogram from device to host using stream 0
        cudaMemcpyAsync(h_histogram, d_histogram, sizeof(unsigned int) * B,
                        cudaMemcpyDeviceToHost, streams[0]);
        cudaStreamSynchronize(streams[0]);

        // Write output - convert to int for file format compatibility
        std::ofstream sol_fs(sol_path, std::ios::binary);
        if (!sol_fs)
        {
            std::cerr << "Error: Could not open output file " << sol_path << std::endl;
            // Cleanup resources before returning
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

        auto int_histogram = std::make_unique<int[]>(B);
        for (int i = 0; i < B; i++)
        {
            int_histogram[i] = static_cast<int>(h_histogram[i]);
        }
        sol_fs.write(reinterpret_cast<const char *>(int_histogram.get()), sizeof(int) * B);
        sol_fs.close();

        // Proper cleanup of all resources
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

        return sol_path;
    }
}