#pragma once

template <typename scalar_t>
__global__ void softmax_kernel10(scalar_t *__restrict__ a, scalar_t *__restrict__ b, int w, int h)
{
    int row = blockIdx.x;
    int ty = threadIdx.y;
    int warp_id = ty / 32;
    int lane_id = ty % 32;

    __shared__ float reduction[BLOCK_DIM_Y / 32];
    float4 reg_array[CEILING((WIDTH / 4), BLOCK_DIM_Y)];

    int reg_array_idx = 0;
    if (row < h)
    {
        float maxval = 0;
#pragma unroll URF
        for (int i = ty; i < WIDTH / 4; i += BLOCK_DIM_Y)
        {
            float4 val = reinterpret_cast<float4 *>(&a[row * WIDTH + i * 4])[0];
            maxval = fmaxf(maxval, val.x);
            maxval = fmaxf(maxval, val.y);
            maxval = fmaxf(maxval, val.z);
            maxval = fmaxf(maxval, val.w);
            reg_array[reg_array_idx] = val;
            reg_array_idx += 1;
        }
        maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, 16, 32));
        maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, 8, 32));
        maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, 4, 32));
        maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, 2, 32));
        maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, 1, 32));

        if (lane_id == 0)
        {
            reduction[warp_id] = maxval;
        }
        __syncthreads();
        if (warp_id == 0)
        {
            maxval = ty < BLOCK_DIM_Y / 32 ? reduction[ty] : 0;
            maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, 16, 32));
            maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, 8, 32));
            maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, 4, 32));
            maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, 2, 32));
            maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, 1, 32));
        }
        if (ty == 0)
        {
            reduction[0] = maxval;
        }
        __syncthreads();
        maxval = reduction[0];
        float divisor = 0.f;
        reg_array_idx = 0;
#pragma unroll URF
        for (int i = ty; i < WIDTH / 4; i += BLOCK_DIM_Y)
        {
            float4 val = reg_array[reg_array_idx];
            val.x = __expf(val.x - maxval);
            val.y = __expf(val.y - maxval);
            val.z = __expf(val.z - maxval);
            val.w = __expf(val.w - maxval);
            divisor += val.x;
            divisor += val.y;
            divisor += val.z;
            divisor += val.w;
            reg_array[reg_array_idx] = val;
            reg_array_idx += 1;
        }

        divisor += __shfl_xor_sync(0xffffffff, divisor, 16, 32);
        divisor += __shfl_xor_sync(0xffffffff, divisor, 8, 32);
        divisor += __shfl_xor_sync(0xffffffff, divisor, 4, 32);
        divisor += __shfl_xor_sync(0xffffffff, divisor, 2, 32);
        divisor += __shfl_xor_sync(0xffffffff, divisor, 1, 32);

        if (lane_id == 0)
        {
            reduction[warp_id] = divisor;
        }

        __syncthreads();
        if (warp_id == 0)
        {
            divisor = ty < BLOCK_DIM_Y / 32 ? reduction[ty] : 0;
            divisor += __shfl_xor_sync(0xffffffff, divisor, 16, 32);
            divisor += __shfl_xor_sync(0xffffffff, divisor, 8, 32);
            divisor += __shfl_xor_sync(0xffffffff, divisor, 4);
            divisor += __shfl_xor_sync(0xffffffff, divisor, 2);
            divisor += __shfl_xor_sync(0xffffffff, divisor, 1);
        }

        if (ty == 0)
        {
            reduction[0] = divisor;
        }

        __syncthreads();
        divisor = reduction[0];

        reg_array_idx = 0;
#pragma unroll URF
        for (int i = ty; i < WIDTH / 4; i += BLOCK_DIM_Y)
        {
            float4 val = reg_array[reg_array_idx];
            val.x = val.x / divisor;
            val.y = val.y / divisor;
            val.z = val.z / divisor;
            val.w = val.w / divisor;
            reinterpret_cast<float4 *>(&b[row * WIDTH + i * 4])[0] = val;
            reg_array_idx += 1;
        }
    }
}