// #pragma once

// #include <atomic>
// #include <cassert>
// #include <cstdio>
// #include <cuda_runtime.h>

// struct shmem_meta_data {
//     unsigned int in_use : 1;
//     unsigned int size : 31;
// } __attribute__((packed));

// /**
//  * @brief Initialize the shared memory region.
//  *
//  * @param size Size of the shared memory region.
//  */
// inline __device__ void init_shmem(size_t size) {
//     if (threadIdx.x == 0) {
//         extern __shared__ char shmem[];
//         size_t *shmem_size_t = (size_t *)shmem;
//         shmem_size_t[0] = size;  // How large is our shared memory region?
//         shmem_size_t[1] = 0b0;   // lowest bit indicates whether we are using the shared memory region
//         for (size_t i = 2; i < size / sizeof(size_t); i++) {
//             shmem_size_t[i] = 0x0;
//         }
//     }
// }

// /**
//  * @brief Allocate memory using the shared memory region.
//  *        Not thread safe.
//  *
//  * @param size  Size of the memory to allocate.
//  * @return void*  Pointer to the allocated memory.
//  */
// inline __device__ void *sh_malloc(size_t requested_size) {

//     printf("requested_size: %ld\n", requested_size);

//     extern __shared__ char shmem[];
//     size_t *shmem_size_t = (size_t *)shmem;
//     size_t max_size = shmem_size_t[0];
//     shmem_meta_data *meta_data = ((shmem_meta_data *)&shmem_size_t[1]);
//     size_t current_offset = sizeof(size_t);

//     // find a free block
//     while (meta_data->in_use == 0b1 && current_offset < max_size) {
//         // go forward to next block.
//         current_offset += sizeof(shmem_meta_data) + meta_data->size;
//         // look at the next block.
//         meta_data = ((shmem_meta_data *)&shmem_size_t[current_offset]);
//     }

//     // we are at the end of the shared memory region, no allocation is possible.
//     if (current_offset + sizeof(shmem_meta_data) + requested_size >= max_size) {
//         // we don't have enough memory
//         assert(!"Not enough memory");
//         return nullptr;
//     }

//     meta_data->size = requested_size;
//     meta_data->in_use = 0b1;

//     // shmem[current_offset + sizeof(shmem_meta_data) + requested_size] = 0x0;
//     // shmem[current_offset + sizeof(shmem_meta_data) + requested_size + 1] = 0x0;
//     // shmem[current_offset + sizeof(shmem_meta_data) + requested_size + 2] = 0x0;
//     // shmem[current_offset + sizeof(shmem_meta_data) + requested_size + 3] = 0x0;

//     return meta_data + 1;
// }

// /**
//  * @brief Free memory allocated using sh_malloc.
//  *
//  * @param ptr Pointer to the memory to free.
//  */
// inline __device__ void sh_free(void *ptr) {
//     // Do nothing for trivial allocator.
//     // just free, nothing else to do, doing no coalescing.
//     // There are LOTS of ways to break this, but we don't care for now.
//     char *ptr_char = (char *)ptr;
//     ptr_char -= sizeof(shmem_meta_data);
//     shmem_meta_data *meta_data = (shmem_meta_data *)ptr_char;
//     meta_data->in_use = 0b0;
// }

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

    shmem_size_t[1] = offset + size;
    // return the pointer to the allocated memory
    return shmem + offset;
}

/**
 * @brief Free memory allocated using sh_malloc.
 *
 * @param ptr Pointer to the memory to free.
 */
inline __device__ void sh_free(void *ptr) {
    // Do nothing for trivial allocator.
}
