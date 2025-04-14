#ifndef MY_BENCHMARK_H
#define MY_BENCHMARK_H

#include <cuda_runtime_api.h>

#include <chrono>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#ifndef CUBLAS_CHECK_AND_EXIT
#define CUBLAS_CHECK_AND_EXIT(error)                                                \
    {                                                                               \
        auto status = static_cast<cublasStatus_t>(error);                           \
        if (status != CUBLAS_STATUS_SUCCESS) {                                      \
            std::cout << status << " " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(status);                                                      \
        }                                                                           \
    }
#endif  // CUBLAS_CHECK_AND_EXIT

#ifndef CUDA_CHECK_AND_EXIT
#define CUDA_CHECK_AND_EXIT(error)                                                                      \
    {                                                                                                   \
        auto status = static_cast<cudaError_t>(error);                                                  \
        if (status != cudaSuccess) {                                                                    \
            std::cout << cudaGetErrorString(status) << " " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(status);                                                                          \
        }                                                                                               \
    }
#endif

namespace benchmark {

struct cuda_event_timer {
    // Returns execution time in ms.
    template<typename Kernel>
    static float execution(Kernel &&kernel, const unsigned int warm_up_runs, const unsigned int runs, cudaStream_t stream) {
        cudaEvent_t startEvent, stopEvent;
        CUDA_CHECK_AND_EXIT(cudaEventCreate(&startEvent));
        CUDA_CHECK_AND_EXIT(cudaEventCreate(&stopEvent));
        CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

        for (unsigned int i = 0; i < warm_up_runs; i++) {
            kernel(stream);
        }

        CUDA_CHECK_AND_EXIT(cudaGetLastError());
        CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

        CUDA_CHECK_AND_EXIT(cudaEventRecord(startEvent, stream));
        for (unsigned int i = 0; i < runs; i++) {
            kernel(stream);
        }
        CUDA_CHECK_AND_EXIT(cudaEventRecord(stopEvent, stream));
        CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

        float time;
        CUDA_CHECK_AND_EXIT(cudaEventElapsedTime(&time, startEvent, stopEvent));
        CUDA_CHECK_AND_EXIT(cudaEventDestroy(startEvent));
        CUDA_CHECK_AND_EXIT(cudaEventDestroy(stopEvent));
        return time / (float)runs;
    }
};

inline double benchmark(uint64_t samples, uint64_t iterations,
                        const std::function<void()> &op, int64_t warm_up_runs = 10) {
    // Warm up the kernel.
    for (uint64_t i = 0; i < warm_up_runs; i++) {
        op();
    }

    double best = std::numeric_limits<double>::infinity();
    for (uint64_t i = 0; i < samples; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        for (uint64_t j = 0; j < iterations; j++) {
            op();
        }
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
        best = std::min(best, elapsed_seconds);
    }
    return best / iterations;
}

};  // namespace benchmark

#endif