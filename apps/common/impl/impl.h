// Adapted from https:github.com/SzymonOzog/FastSoftmax.git

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

template<typename T>
__device__ void
exp_matrix(const T &input,
           T &output) {
    auto &input_data = input.array;
    auto &output_data = output.array;
    constexpr int64_t num_row = input.rows;
    constexpr int64_t num_col = input.cols_by_4;
#pragma unroll URF
    for (int m = 0; m < num_row * num_col; m++) {
        float4 val = input_data[m];
        val.x = __expf(val.x);
        val.y = __expf(val.y);
        val.z = __expf(val.z);
        val.w = __expf(val.w);
        output_data[m] = val;
    }
}

template<typename T1, typename T2>
__device__ void
divide_vec(T1 &vec,
           const T2 &input,
           T2 &output) {
    auto &in_data = input.array;
    auto &out_data = output.array;
    auto &vec_data = vec.val;

    constexpr int64_t num_row = input.rows;
    constexpr int64_t num_col = input.cols_by_4;
    static_assert(vec.size == num_row);

    for (int m = 0; m < num_row; m++) {
        float div_val = vec_data[m];
#pragma unroll URF
        for (int i = 0; i < num_col; i++) {
            float4 val = in_data[i];
            val.x = (val.x / div_val);
            val.y = (val.y / div_val);
            val.z = (val.z / div_val);
            val.w = (val.w / div_val);
            out_data[i] = val;
        }
    }
}

template<typename T1, typename T2>
__device__ void
subtract_vec(T1 &vec,
             const T2 &input,
             T2 &output) {
    auto &in_data = input.array;
    auto &out_data = output.array;
    auto &vec_data = vec.val;

    constexpr int64_t num_row = input.rows;
    constexpr int64_t num_col = input.cols_by_4;
    static_assert(vec.size == num_row);

    for (int m = 0; m < num_row; m++) {
        float sub_val = vec_data[m];
#pragma unroll URF
        for (int i = 0; i < num_col; i++) {
            float4 val = in_data[i];
            val.x = (val.x - sub_val);
            val.y = (val.y - sub_val);
            val.z = (val.z - sub_val);
            val.w = (val.w - sub_val);
            out_data[i] = val;
        }
    }
}

template<int stride,
         typename T1,
         typename T2>
__device__ void
sum_row(T1 &output, const T2 &input) {
    constexpr int64_t num_row = input.rows;
    constexpr int64_t num_col = input.cols_by_4;
    static_assert(output.size == num_row);

    float sum = 0;
    int ty = threadIdx.y;
    int warp_id = ty / 32;
    int reg_array_idx = 0;
    int lane_id = ty % 32;
    __shared__ float reduction[stride / 32];
    auto &reg_array = input.array;

#pragma unroll URF
    for (int r = 0; r < num_row; r++) {
        for (int i = 0; i < num_col; i++) {
            float4 val = reg_array[reg_array_idx];
            sum += val.x;
            sum += val.y;
            sum += val.z;
            sum += val.w;
            reg_array_idx += 1;
        }
        sum += __shfl_xor_sync(0xffffffff, sum, 16, 32);
        sum += __shfl_xor_sync(0xffffffff, sum, 8, 32);
        sum += __shfl_xor_sync(0xffffffff, sum, 4, 32);
        sum += __shfl_xor_sync(0xffffffff, sum, 2, 32);
        sum += __shfl_xor_sync(0xffffffff, sum, 1, 32);

        if (lane_id == 0) {
            reduction[warp_id] = sum;
        }
        __syncthreads();
        if (warp_id == 0) {
            sum = ty < stride / 32 ? reduction[ty] : 0;
            sum += __shfl_xor_sync(0xffffffff, sum, 16, 32);
            sum += __shfl_xor_sync(0xffffffff, sum, 8, 32);
            sum += __shfl_xor_sync(0xffffffff, sum, 4);
            sum += __shfl_xor_sync(0xffffffff, sum, 2);
            sum += __shfl_xor_sync(0xffffffff, sum, 1);
        }
        if (ty == 0) {
            reduction[0] = sum;
        }
        __syncthreads();
        sum = reduction[0];

        output.val[r] = sum;
    }
}

template<int stride,
         typename T1,
         typename T2>
__device__ void
max_shuffle(T1 &output, const T2 &input) {
    constexpr int64_t num_row = input.rows;
    constexpr int64_t num_col = input.cols_by_4;
    static_assert(output.size == num_row);

    float maxval = 0;
    int ty = threadIdx.y;
    int warp_id = ty / 32;
    int reg_array_idx = 0;
    int lane_id = ty % 32;
    __shared__ float reduction[stride / 32];
    auto &reg_array = input.array;

#pragma unroll URF
    for (int r = 0; r < num_row; r++) {
        for (int i = 0; i < num_col; i++) {
            float4 val = reg_array[reg_array_idx];
            maxval = fmaxf(maxval, val.x);
            maxval = fmaxf(maxval, val.y);
            maxval = fmaxf(maxval, val.z);
            maxval = fmaxf(maxval, val.w);
            reg_array_idx += 1;
        }
        maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, 16, 32));
        maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, 8, 32));
        maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, 4, 32));
        maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, 2, 32));
        maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, 1, 32));

        if (lane_id == 0) {
            reduction[warp_id] = maxval;
        }
        __syncthreads();
        if (warp_id == 0) {
            maxval = ty < stride / 32 ? reduction[ty] : 0;
            maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, 16, 32));
            maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, 8, 32));
            maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, 4, 32));
            maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, 2, 32));
            maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, 1, 32));
        }
        if (ty == 0) {
            reduction[0] = maxval;
        }
        __syncthreads();
        maxval = reduction[0];

        output.val[r] = maxval;
    }
}

template<int64_t stride, typename T1, typename T2>
__device__ void
blur_x(const T1 &input,
       T2 &output) {
    // Treat as normal float arrays.
    const float *input_data = reinterpret_cast<const float *>(input.array);
    float *output_data = reinterpret_cast<float *>(output.array);
    constexpr int64_t num_row = output.rows;
    constexpr int64_t num_col = output.cols_by_4 * 4;
    constexpr int64_t input_num_col = input.cols_by_4 * 4;

#pragma unroll URF

    for (int m = 0; m < num_row; m++) {
        for (int n = 0; n < num_col; n++) {
            float sum = 0;
            for (int s = 0; s < stride; s++) {
                sum += input_data[s + n];
            }
            output_data[n] = sum / stride;
        }
        output_data += num_col;
        input_data += input_num_col;
    }
}