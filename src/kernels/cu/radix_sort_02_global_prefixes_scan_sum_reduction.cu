#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"
#include "../kernels.h"

namespace cuda {
void radix_sort_02_global_prefixes_scan_sum_reduction(
    const gpu::WorkSize &workSize_level0,
    const gpu::WorkSize &workSize_level1,
    const gpu::gpu_mem_32u &local_counts,
    gpu::gpu_mem_32u &block_prefixes,
    gpu::gpu_mem_32u &group_sums,
    gpu::gpu_mem_32u &group_prefixes,
    gpu::gpu_mem_32u &top_sums,
    unsigned int digit,
    unsigned int padded_blocks)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());

    const size_t digit_offset = (size_t) digit * padded_blocks;
    gpu::gpu_mem_32u counts_slice(local_counts, digit_offset);
    gpu::gpu_mem_32u prefixes_slice(block_prefixes, digit_offset);

    cuda::prefix_sum_01_reduction(workSize_level0, counts_slice, prefixes_slice, group_sums);

    cuda::prefix_sum_01_reduction(workSize_level1, group_sums, group_prefixes, top_sums);
}
} // namespace cuda
