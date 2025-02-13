#pragma once

#include "adt.h"
#include <cstdio>
#include <cub/block/block_reduce.cuh>
#include <cuda.h>
#include <cuda/atomic>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

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
