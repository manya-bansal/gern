#ifndef MY_BENCHMARK_H
#define MY_BENCHMARK_H

#include <cuda_runtime_api.h>

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

namespace input_gen {

void fillArrayWithRandomNumbers(float *arr, int M, float min = 0.0f, float max = 1.0f) {
    // Create a random device and a Mersenne Twister engine
    std::random_device rd;
    std::mt19937 gen(rd());

    // Create a distribution to generate random numbers in the range [min, max]
    std::uniform_real_distribution<float> dis(min, max);

    // Fill the array with random numbers
    for (int i = 0; i < M; ++i) {
        arr[i] = dis(gen);
    }
}

}  // namespace input_gen

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