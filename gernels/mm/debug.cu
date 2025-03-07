#include "/home/manya/gern/gernels/mm/impl/sh_malloc.h"
#include "cassert"
#include "impl/matrix-gpu.h"
#include "impl/matrix_multiply.h"
#include <cuda_runtime.h>

template<int64_t block_x, int64_t block_y, int64_t k_dim, int64_t k_tiled, int64_t one_val, int64_t thread_x>
__global__ void function_55(impl::MatrixGPU<8, 24, 24, 1> A, impl::MatrixGPU<24, 8, 8, 1> B, impl::MatrixGPU<8, 8, 8, 1> C, int64_t smem_size) {

    init_shmem(smem_size);

    int64_t _gern_i_1_7_13_19_25_31_37_43_49 = ((blockIdx.y * block_x) + 0);
    int64_t _gern_j_2_8_14_20_26_32_38_44 = ((blockIdx.x * block_y) + 0);
    int64_t _gern_j_2_8_14_20_26_32_38 = _gern_j_2_8_14_20_26_32_38_44;
    constexpr int64_t _gern_tj_5_11_17_23_29_35_41 = block_y;
    int64_t _gern_i_1_7_13_19_25_31_37 = _gern_i_1_7_13_19_25_31_37_43_49;
    constexpr int64_t _gern_ti_4_10_16_22_28_34_40 = block_x;

    auto _query_C_56 = C.template query_global_2_global<_gern_ti_4_10_16_22_28_34_40, _gern_tj_5_11_17_23_29_35_41>(_gern_i_1_7_13_19_25_31_37, _gern_j_2_8_14_20_26_32_38);

    for (int64_t _gern_k_3_9_15_21_27_33_39 = 0; (_gern_k_3_9_15_21_27_33_39 < k_dim); _gern_k_3_9_15_21_27_33_39 = (_gern_k_3_9_15_21_27_33_39 + k_tiled)) {

        int64_t _gern_j_2_8_14_20_26_32 = _gern_j_2_8_14_20_26_32_38_44;
        constexpr int64_t _gern_tj_5_11_17_23_29_35 = block_y;
        int64_t _gern_i_1_7_13_19_25_31 = _gern_i_1_7_13_19_25_31_37_43_49;
        constexpr int64_t _gern_ti_4_10_16_22_28_34 = block_x;

        int64_t _gern_k_3_9_15_21_27_33 = _gern_k_3_9_15_21_27_33_39;
        constexpr int64_t _gern_tk_6_12_18_24_30_36 = k_tiled;

        auto _query_A_57 = A.template stage_into_smem<_gern_ti_4_10_16_22_28_34, _gern_tk_6_12_18_24_30_36>(_gern_i_1_7_13_19_25_31, _gern_k_3_9_15_21_27_33);

        int64_t _gern_j_2_8_14_20_26 = _gern_j_2_8_14_20_26_32_38_44;
        constexpr int64_t _gern_tj_5_11_17_23_29 = block_y;
        int64_t _gern_i_1_7_13_19_25 = _gern_i_1_7_13_19_25_31_37_43_49;
        constexpr int64_t _gern_ti_4_10_16_22_28 = block_x;

        int64_t _gern_k_3_9_15_21_27 = _gern_k_3_9_15_21_27_33_39;
        constexpr int64_t _gern_tk_6_12_18_24_30 = k_tiled;

        auto _query_B_58 = B.template stage_into_smem<_gern_tk_6_12_18_24_30, _gern_tj_5_11_17_23_29>(_gern_k_3_9_15_21_27, _gern_j_2_8_14_20_26);

        int64_t _gern_i_1_7_13_19 = ((threadIdx.x * thread_x) + 0);
        int64_t _gern_j_2_8_14 = ((threadIdx.y * thread_x) + 0);
        int64_t _gern_j_2_8 = (_gern_j_2_8_14 + _gern_j_2_8_14_20_26_32_38_44);
        constexpr int64_t _gern_tj_5_11 = thread_x;
        int64_t _gern_i_1_7 = (_gern_i_1_7_13_19 + _gern_i_1_7_13_19_25_31_37_43_49);
        constexpr int64_t _gern_ti_4_10 = thread_x;

        auto _query_C_59 = _query_C_56.template query_global_2_global<_gern_ti_4_10, _gern_tj_5_11>(_gern_i_1_7_13_19, _gern_j_2_8_14);

        for (int64_t _gern_k_3_9 = 0; (_gern_k_3_9 < k_tiled); _gern_k_3_9 = (_gern_k_3_9 + one_val)) {

            int64_t _gern_j_2 = (_gern_j_2_8_14 + _gern_j_2_8_14_20_26_32_38_44);
            constexpr int64_t _gern_tj_5 = thread_x;
            int64_t _gern_i_1 = (_gern_i_1_7_13_19 + _gern_i_1_7_13_19_25_31_37_43_49);
            constexpr int64_t _gern_ti_4 = thread_x;

            int64_t _gern_k_3 = (_gern_k_3_9 + _gern_k_3_9_15_21_27_33_39);
            constexpr int64_t _gern_tk_6 = one_val;

            auto _query_A_60 = _query_A_57.template stage_into_smem<_gern_ti_4, _gern_tk_6>(_gern_i_1_7_13_19, _gern_k_3_9);

            auto _query_B_61 = _query_B_58.template stage_into_smem<_gern_tk_6, _gern_tj_5>(_gern_k_3_9, _gern_j_2_8_14);

            matrix_multiply<one_val>(_query_A_60, _query_B_61, _query_C_59);

            _query_A_60.free_smem();
            _query_B_61.free_smem();
        }

        _query_B_58.free_smem();

        _query_A_57.free_smem();
    }
}

int main() {
    constexpr int64_t block_x = 8;
    constexpr int64_t block_y = 8;
    constexpr int64_t k_dim = 24;
    constexpr int64_t k_tiled = 8;
    constexpr int64_t one_val = 8;
    constexpr int64_t thread_x = 1;

    impl::MatrixGPU<8, 24, 24, 1> A;
    impl::MatrixGPU<24, 8, 8, 1> B;
    impl::MatrixGPU<8, 8, 8, 1> C;

    A.vvals(1.0f);
    B.vvals(1.0f);
    C.vvals(0.0f);

    int64_t smem_size = 8 * 8 * 4 * 2 + 640 + 2;

    dim3 grid_62 = dim3((((block_y + (C.col - 0)) - 1) / block_y), (((block_x + (C.row - 0)) - 1) / block_x), 1);
    dim3 block_63 = dim3((((thread_x + (block_x - 0)) - 1) / thread_x), (((thread_x + (block_y - 0)) - 1) / thread_x), 1);

    auto function_sp_64 = function_55<block_x, block_y, k_dim, k_tiled, one_val, thread_x>;
    CUDA_CHECK(cudaFuncSetAttribute(function_sp_64, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    function_sp_64<<<grid_62, block_63, smem_size>>>(A, B, C, smem_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto C_cpu = C.get();
    for (int64_t i = 0; i < 8; i++) {
        for (int64_t j = 0; j < 8; j++) {
            std::cout << C_cpu(i, j) << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
