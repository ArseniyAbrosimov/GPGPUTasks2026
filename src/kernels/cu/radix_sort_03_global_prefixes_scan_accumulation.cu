#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"
#include "../kernels.h"

__global__ void radix_sort_03_global_prefixes_scan_accumulation(
    const unsigned int* block_prefixes,
          unsigned int* bucket_offsets,
    unsigned int digit,
    unsigned int padded_blocks)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;

    const unsigned int bucket_total = block_prefixes[padded_blocks - 1];

    if (digit == 0)
        bucket_offsets[0] = 0;

    bucket_offsets[digit + 1] = bucket_offsets[digit] + bucket_total;
}

namespace cuda {
void radix_sort_03_global_prefixes_scan_accumulation(
    const gpu::WorkSize &workSize_level0,
    const gpu::WorkSize &workSize_level1,
    gpu::gpu_mem_32u &block_prefixes,
    gpu::gpu_mem_32u &group_prefixes,
    const gpu::gpu_mem_32u &top_sums,
    gpu::gpu_mem_32u &top_acc,
    gpu::gpu_mem_32u &bucket_offsets,
    unsigned int digit,
    unsigned int padded_blocks)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();

    cuda::prefix_sum_02_prefix_accumulation(workSize_level1, top_sums, top_acc, group_prefixes);

    const size_t digit_offset = (size_t) digit * padded_blocks;
    gpu::gpu_mem_32u prefixes_slice(block_prefixes, digit_offset);
    cuda::add_upper(workSize_level0, prefixes_slice, group_prefixes);

    ::radix_sort_03_global_prefixes_scan_accumulation<<<1, 1, 0, stream>>>(
        block_prefixes.cuptr() + digit_offset, bucket_offsets.cuptr(), digit, padded_blocks);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
