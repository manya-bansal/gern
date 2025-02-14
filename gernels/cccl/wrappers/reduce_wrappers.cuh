#pragma once

#include "adt.h"
#include <cstdio>
#include <cub/block/block_reduce.cuh>
#include <cuda.h>
#include <cuda/atomic>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

template<int block_size,
         typename T1,
         typename T2>
__device__ void block_reduce(T1 &output,
                             const T2 &input) {

    using BlockReduce = cub::BlockReduce<float, block_size>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    static_assert(input.size == output.size * block_size);

    const float *input_data = input.data;
    for (int i = 0; i < output.size; i++) {
        int thread_idx = threadIdx.x;
        float sum = BlockReduce(temp_storage).Sum(input_data[thread_idx]);
        if (thread_idx == 0) {
            output.data[i] = sum;
        }
        input_data += block_size;
    }
}

__device__ void global_sum_single(float *total_sum,
                                  float input) {
    if (threadIdx.x == 0) {
        cuda::atomic_ref<float, cuda::thread_scope_device> atomic_result(*total_sum);
        atomic_result.fetch_add(input,
                                cuda::memory_order_relaxed);
    }
}

template<int k,
         int block_size,
         typename T2>
__device__ void grid_reduce_total(float *total_sum,
                                  const T2 &input) {

    using BlockReduce = cub::BlockReduce<float, block_size>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int i = threadIdx.x;
    float internal_sum = input.data[i];
    float sum = BlockReduce(temp_storage).Sum(internal_sum);
    // global_sum_single(total_sum, sum);

    if (threadIdx.x == 0) {
        cuda::atomic_ref<float, cuda::thread_scope_device> atomic_result(*total_sum);
        atomic_result.fetch_add(sum,
                                cuda::memory_order_relaxed);
    }
}

template<int k, typename T>
__device__ void global_sum(float *total_sum,
                           const T &input) {
    if (threadIdx.x == 0) {
        cuda::atomic_ref<float, cuda::thread_scope_device> atomic_result(*total_sum);
        for (int i = 0; i < k; i++) {
            atomic_result.fetch_add(input.data[i],
                                    cuda::memory_order_relaxed);
        }
    }
}
