#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void matrix_transpose_coalesced_via_local_memory(
                       const float* matrix,            // w x h
                             float* transposed_matrix, // h x w
                             unsigned int w,
                             unsigned int h)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    __shared__ float workgroup_data[GROUP_SIZE_Y * (GROUP_SIZE_X + 1)];
    if (x < w && y < h)
        workgroup_data[threadIdx.y * (GROUP_SIZE_X + 1) + threadIdx.x] = matrix[y * w + x];
    __syncthreads();


    const unsigned int y_t = blockIdx.x * blockDim.x + threadIdx.y;
    const unsigned int x_t = blockIdx.y * blockDim.y + threadIdx.x;
    if (y_t < w && x_t < h)
        transposed_matrix[y_t * h + x_t] = workgroup_data[threadIdx.x * (GROUP_SIZE_X + 1) + threadIdx.y];
}

namespace cuda {
void matrix_transpose_coalesced_via_local_memory(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &matrix, gpu::gpu_mem_32f &transposed_matrix, unsigned int w, unsigned int h)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_transpose_coalesced_via_local_memory<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(matrix.cuptr(), transposed_matrix.cuptr(), w, h);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
