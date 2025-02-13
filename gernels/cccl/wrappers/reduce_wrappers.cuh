#include <cstdio>
#include <cub/block/block_reduce.cuh>
#include <cuda/atomic>
#include <thrust/device_vector.h>

// This is an atomic sum.
template<typename T, int k>
__device__ void global_sum(int *total_sum,
                           const T &input) {
    if (threadIdx.x == 0) {
        cuda::atomic_ref<int, cuda::thread_scope_device> atomic_result(*total_sum);
        for (int i = 0; i < k; i++) {
            atomic_result.fetch_add(input.data[i],
                                    cuda::memory_order_relaxed);
        }
    }
}
