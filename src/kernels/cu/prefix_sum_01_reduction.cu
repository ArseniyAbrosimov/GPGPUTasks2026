#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void prefix_sum_01_reduction(const unsigned int* pow2_sum,
                                  unsigned int* next_pow2_sum,
                                  unsigned int* block_sum)
{
    __shared__ unsigned int workgroup_data[GROUP_SIZE];

    const unsigned int local_id = threadIdx.x;
    const unsigned int index = blockIdx.x * GROUP_SIZE + local_id;

    workgroup_data[local_id] = pow2_sum[index];
    __syncthreads();

    for (unsigned int offset = 1; offset < GROUP_SIZE; offset <<= 1) {
        const unsigned int pref = (local_id >= offset) ? workgroup_data[local_id - offset] : 0;
        __syncthreads();
        workgroup_data[local_id] += pref;
        __syncthreads();
    }

    next_pow2_sum[index] = workgroup_data[local_id];

    if (local_id == GROUP_SIZE - 1)
        block_sum[blockIdx.x] = workgroup_data[local_id];
}

namespace cuda {
void prefix_sum_01_reduction(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &pow2_sum, gpu::gpu_mem_32u &next_pow2_sum, gpu::gpu_mem_32u &block_sum)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefix_sum_01_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(pow2_sum.cuptr(), next_pow2_sum.cuptr(), block_sum.cuptr());
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
