
#ifndef MY_BENCHMARK_H
#define MY_BENCHMARK_H

#include <algorithm>
#include <cassert>
#include <chrono>
#include <functional>
#include <limits>

namespace my_benchmark {

// This benchmarking code is stolen from Halide :) !!!!

// Prefer high_resolution_clock, but only if it's steady...
template<bool HighResIsSteady = std::chrono::high_resolution_clock::is_steady>
struct SteadyClock {
    using type = std::chrono::high_resolution_clock;
};

// ...otherwise use steady_clock.
template<>
struct SteadyClock<false> {
    using type = std::chrono::steady_clock;
};

inline SteadyClock<>::type::time_point benchmark_now() {
    return SteadyClock<>::type::now();
}

inline double benchmark_duration_seconds(SteadyClock<>::type::time_point start,
                                         SteadyClock<>::type::time_point end) {
    return std::chrono::duration_cast<std::chrono::duration<double>>(end - start)
        .count();
}

inline double benchmark(uint64_t samples, uint64_t iterations,
                        const std::function<void()> &op) {
    double best = std::numeric_limits<double>::infinity();
    for (uint64_t i = 0; i < samples; i++) {
        auto start = benchmark_now();
        for (uint64_t j = 0; j < iterations; j++) {
            op();
        }
        auto end = benchmark_now();
        double elapsed_seconds = benchmark_duration_seconds(start, end);
        best = std::min(best, elapsed_seconds);
    }
    return best / iterations;
}

inline double benchmark_return(uint64_t samples, uint64_t iterations,
                               const std::function<double()> &op) {
    double best = std::numeric_limits<double>::infinity();
    for (uint64_t i = 0; i < samples; i++) {
        double elapsed_seconds = 0;
        for (uint64_t j = 0; j < iterations; j++) {
            elapsed_seconds += op();
        }
        best = std::min(best, elapsed_seconds);
    }
    return best / iterations;
}

typedef void (*callback_function)(void);
inline double benchmark_new(uint64_t samples, uint64_t iterations,
                            callback_function op) {
    double best = std::numeric_limits<double>::infinity();
    for (uint64_t i = 0; i < samples; i++) {
        auto start = benchmark_now();
        for (uint64_t j = 0; j < iterations; j++) {
            op();
        }
        auto end = benchmark_now();
        double elapsed_seconds = benchmark_duration_seconds(start, end);
        best = std::min(best, elapsed_seconds);
    }
    return best / iterations;
}

}  // namespace my_benchmark

#include <cuda_runtime.h>
#include <iostream>
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

struct measure {
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

};  // namespace benchmark

#endif