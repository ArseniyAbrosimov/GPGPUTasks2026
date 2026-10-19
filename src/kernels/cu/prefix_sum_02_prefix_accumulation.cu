#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void prefix_sum_02_prefix_accumulation(const unsigned int* block_sum,
                                      unsigned int* block_acc,
                                unsigned int num_blocks)
{
    __shared__ unsigned int blocks_prefix[GROUP_SIZE];

    const unsigned int local_id = threadIdx.x;
    
    const unsigned int per_thread = (num_blocks + GROUP_SIZE - 1) / GROUP_SIZE;

    const unsigned int begin = local_id * per_thread;
    const unsigned int end = (begin + per_thread < num_blocks) ? (begin + per_thread) : num_blocks;

    unsigned int blocks_sum = 0;
    for (unsigned int i = begin; i < end; ++i)
        blocks_sum += block_sum[i];

    blocks_prefix[local_id] = blocks_sum;
    __syncthreads();



    for (unsigned int offset = 1; offset < GROUP_SIZE; offset <<= 1) {
        const unsigned int pref = (local_id >= offset) ? blocks_prefix[local_id - offset] : 0;
        __syncthreads();
        blocks_prefix[local_id] += pref;
        __syncthreads();
    }



    
    unsigned int acc = blocks_prefix[local_id] - blocks_sum;
    for (unsigned int i = begin; i < end; ++i) {
        block_acc[i] = acc;
        acc += block_sum[i];
    }
}

__global__ void add_block_acc(unsigned int* next_pow2_sum,
                                  const unsigned int* block_acc)
{
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    next_pow2_sum[gid] += block_acc[blockIdx.x];
}

__global__ void add_upper(unsigned int* prefix,
                          const unsigned int* upper)
{
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x > 0)
        prefix[gid] += upper[blockIdx.x - 1];
}

namespace cuda {
void prefix_sum_02_prefix_accumulation(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &block_sum, gpu::gpu_mem_32u &block_acc, gpu::gpu_mem_32u &next_pow2_sum)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();

    const unsigned int num_blocks = block_sum.number();
    ::prefix_sum_02_prefix_accumulation<<<1, workSize.cuBlockSize(), 0, stream>>>(block_sum.cuptr(), block_acc.cuptr(), num_blocks);
    CUDA_CHECK_KERNEL(stream);

    ::add_block_acc<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(next_pow2_sum.cuptr(), block_acc.cuptr());
    CUDA_CHECK_KERNEL(stream);
}

void add_upper(const gpu::WorkSize &workSize, gpu::gpu_mem_32u &prefix, const gpu::gpu_mem_32u &upper)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::add_upper<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(prefix.cuptr(), upper.cuptr());
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
