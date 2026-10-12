#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void matrix_multiply_via_local_memory(
                       const float* a, // rows=h x cols=k
                       const float* b, // rows=k x cols=w
                             float* c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    const unsigned int x = blockIdx.x * GROUP_SIZE_X + threadIdx.x;
    const unsigned int y = blockIdx.y * GROUP_SIZE_X + threadIdx.y;

    __shared__ float workgroup_data_a[GROUP_SIZE];
    __shared__ float workgroup_data_b[GROUP_SIZE];

    float mul = 0.0f;
    for (unsigned int i = 0; i < k; i += GROUP_SIZE_X) {
            workgroup_data_a[threadIdx.y * GROUP_SIZE_X + threadIdx.x] = a[y * k + i + threadIdx.x];
            workgroup_data_b[threadIdx.y * GROUP_SIZE_X + threadIdx.x] = b[(i + threadIdx.y) * w + x];
        __syncthreads();

        for (unsigned int j = 0; j < GROUP_SIZE_X; ++j)
            mul += workgroup_data_a[threadIdx.y * GROUP_SIZE_X + j] * workgroup_data_b[j * GROUP_SIZE_X + threadIdx.x];

        __syncthreads();
    }

    if (x < w && y < h)
        c[y * w + x] = mul;
}

namespace cuda {
void matrix_multiply_via_local_memory(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &a, const gpu::gpu_mem_32f &b, gpu::gpu_mem_32f &c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_multiply_via_local_memory<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
