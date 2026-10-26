#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_04_scatter(
    const unsigned int* values,
    const unsigned int* block_prefixes,
          unsigned int* output,
    const unsigned int* bucket_offsets,
    unsigned int n,
    unsigned int shift,
    unsigned int digit,
    unsigned int padded_blocks)
{
    __shared__ unsigned int prefix[GROUP_SIZE];

    const unsigned int local_id = threadIdx.x;
    const unsigned int block = blockIdx.x;
    const unsigned int index = block * GROUP_SIZE + local_id;

    unsigned int value = 0;
    if (index < n)
        value = values[index];

    const bool match = (index < n) && (((value >> shift) & (RADIX_BUCKETS - 1)) == digit);

    prefix[local_id] = match ? 1u : 0u;
    __syncthreads();

    for (unsigned int offset = 1; offset < GROUP_SIZE; offset <<= 1) {
        const unsigned int add = (local_id >= offset) ? prefix[local_id - offset] : 0u;
        __syncthreads();
        prefix[local_id] += add;
        __syncthreads();
    }

    if (!match)
        return;

    const unsigned int rank = prefix[local_id] - 1u;

    const unsigned int block_offset = (block > 0)
        ? block_prefixes[digit * padded_blocks + block - 1]
        : 0u;

    curassert(bucket_offsets[digit] + block_offset + rank < n, 7453404001);

    output[bucket_offsets[digit] + block_offset + rank] = value;
}

namespace cuda {
void radix_sort_04_scatter(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &values, const gpu::gpu_mem_32u &block_prefixes, gpu::gpu_mem_32u &output,
    const gpu::gpu_mem_32u &bucket_offsets, unsigned int n, unsigned int shift, unsigned int digit,
    unsigned int padded_blocks)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_04_scatter<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        values.cuptr(), block_prefixes.cuptr(), output.cuptr(), bucket_offsets.cuptr(),
        n, shift, digit, padded_blocks);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
