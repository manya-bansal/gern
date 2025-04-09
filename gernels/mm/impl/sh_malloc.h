#pragma once

#include <atomic>
#include <cassert>
#include <cuda_runtime.h>

/**
 * @brief Initialize the shared memory region.
 *
 * @param size Size of the shared memory region.
 */
inline __device__ void init_shmem(size_t size) {
    if (threadIdx.x == 0) {
        extern __shared__ char shmem[];
        size_t *shmem_size_t = (size_t *)shmem;
        shmem_size_t[0] = size;                // How large is our shared memory region?
        shmem_size_t[1] = 2 * sizeof(size_t);  // Sets up the offset for the next allocation
    }
}

/**
 * @brief Allocate memory using the shared memory region.
 *        Not thread safe.
 *
 * @param size  Size of the memory to allocate.
 * @return void*  Pointer to the allocated memory.
 */
inline __device__ void *sh_malloc(size_t size) {

    extern __shared__ char shmem[];
    size_t *shmem_size_t = (size_t *)shmem;
    size_t max_size = shmem_size_t[0];
    size_t offset = shmem_size_t[1];

    if (offset + size > max_size) {
        shmem_size_t[1] = sizeof(size_t) * 2;
        return sh_malloc(size);
    }

    shmem_size_t[1] = (offset + size) % max_size;
    // return the pointer to the allocated memory
    return shmem + offset;
}

/**
 * @brief Free memory allocated using sh_malloc.
 *
 * @param ptr Pointer to the memory to free.
 */
inline __device__ void sh_free(void *ptr) {
    __syncthreads();
    // Do nothing for trivial allocator.
}
