#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_01_local_counting(
    const unsigned int* values,
          unsigned int* local_counts,
    unsigned int n,
    unsigned int shift,
    unsigned int padded_blocks)
{
    __shared__ unsigned int counts[RADIX_BUCKETS];

    const unsigned int local_id = threadIdx.x;
    const unsigned int block = blockIdx.x;
    const unsigned int index = block * GROUP_SIZE + local_id;

    if (local_id < RADIX_BUCKETS)
        counts[local_id] = 0;
    __syncthreads();

    if (index < n) {
        const unsigned int digit = (values[index] >> shift) & (RADIX_BUCKETS - 1);
        atomicAdd(&counts[digit], 1u);
    }
    __syncthreads();

    if (local_id < RADIX_BUCKETS) {
        local_counts[local_id * padded_blocks + block] = counts[local_id];
    }
}

namespace cuda {
void radix_sort_01_local_counting(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &values, gpu::gpu_mem_32u &local_counts, unsigned int n, unsigned int shift,
            unsigned int padded_blocks)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();

    ::radix_sort_01_local_counting<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        values.cuptr(), local_counts.cuptr(), n, shift, padded_blocks);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
