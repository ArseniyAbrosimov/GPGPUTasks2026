#pragma once

#include <libgpu/vulkan/engine.h>

namespace cuda {
void aplusb(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n);
void fill_buffer_with_zeros(const gpu::WorkSize &workSize, gpu::gpu_mem_32u &buffer, unsigned int n);

void prefix_sum_01_reduction(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &pow2_sum, gpu::gpu_mem_32u &next_pow2_sum, gpu::gpu_mem_32u &block_sum);
void prefix_sum_02_prefix_accumulation(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &block_sum, gpu::gpu_mem_32u &block_acc, gpu::gpu_mem_32u &next_pow2_sum);
void add_upper(const gpu::WorkSize &workSize, gpu::gpu_mem_32u &prefix, const gpu::gpu_mem_32u &upper);

void radix_sort_01_local_counting(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &values, gpu::gpu_mem_32u &local_counts, unsigned int n, unsigned int shift,
    unsigned int padded_blocks);
void radix_sort_02_global_prefixes_scan_sum_reduction(const gpu::WorkSize &workSize_level0, const gpu::WorkSize &workSize_level1,
    const gpu::gpu_mem_32u &local_counts, gpu::gpu_mem_32u &block_prefixes,
    gpu::gpu_mem_32u &group_sums, gpu::gpu_mem_32u &group_prefixes, gpu::gpu_mem_32u &top_sums,
    unsigned int digit, unsigned int padded_blocks);
void radix_sort_03_global_prefixes_scan_accumulation(const gpu::WorkSize &workSize_level0, const gpu::WorkSize &workSize_level1,
    gpu::gpu_mem_32u &block_prefixes, gpu::gpu_mem_32u &group_prefixes, const gpu::gpu_mem_32u &top_sums,
    gpu::gpu_mem_32u &top_acc, gpu::gpu_mem_32u &bucket_offsets, unsigned int digit, unsigned int padded_blocks);
void radix_sort_04_scatter(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &values, const gpu::gpu_mem_32u &block_prefixes, gpu::gpu_mem_32u &output,
    const gpu::gpu_mem_32u &bucket_offsets, unsigned int n, unsigned int shift, unsigned int digit,
    unsigned int padded_blocks);
}

namespace ocl {
const ProgramBinaries& getAplusB();

const ProgramBinaries& getFillBufferWithZeros();
const ProgramBinaries& getRadixSort01LocalCounting();
const ProgramBinaries& getRadixSort02GlobalPrefixesScanSumReduction();
const ProgramBinaries& getRadixSort03GlobalPrefixesScanAccumulation();
const ProgramBinaries& getRadixSort04Scatter();
}

namespace avk2 {
const ProgramBinaries& getAplusB();

const ProgramBinaries& getFillBufferWithZeros();
const ProgramBinaries& getRadixSort01LocalCounting();
const ProgramBinaries& getRadixSort02GlobalPrefixesScanSumReduction();
const ProgramBinaries& getRadixSort03GlobalPrefixesScanAccumulation();
const ProgramBinaries& getRadixSort04Scatter();
}
