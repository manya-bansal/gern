#include "/home/manya/gern/gernels/mm/impl/sh_malloc.h"
#include "cassert"
#include "impl/matrix-gpu.h"
#include "impl/matrix_multiply.h"
#include <chrono>
#include <cuda_runtime.h>
template<int64_t block_x, int64_t block_y, int64_t k_dim, int64_t k_tiled, int64_t thread_k, int64_t thread_x, int64_t thread_y, int64_t warp_x, int64_t warp_y>
__global__ void function_97(impl::MatrixGPU<1024, 1024, 1024, 1> A, impl::MatrixGPU<1024, 1024, 1024, 1> B, impl::MatrixGPU<1024, 1024, 1024, 1> C, int64_t smem_size) {

    init_shmem(smem_size);

    int64_t _gern_i_1_7_13_19_25_31_37_43_49_55_61_67_73_79_85_91 = ((((blockIdx.x / 1) % (((block_x + (C.row - 0)) - 1) / block_x)) * block_x) + 0);
    int64_t _gern_j_2_8_14_20_26_32_38_44_50_56_62_68_74_80_86 = ((((blockIdx.y / 1) % (((block_y + (C.col - 0)) - 1) / block_y)) * block_y) + 0);
    auto _query_C_98 = C.template query_global_2_global<block_x, block_y>(_gern_i_1_7_13_19_25_31_37_43_49_55_61_67_73_79_85_91, (_gern_j_2_8_14_20_26_32_38_44_50_56_62_68_74_80_86 + 0));

    for (int64_t _gern_k_3_9_15_21_27_33_39_45_51_57_63_69_75_81 = 0; (_gern_k_3_9_15_21_27_33_39_45_51_57_63_69_75_81 < k_dim); _gern_k_3_9_15_21_27_33_39_45_51_57_63_69_75_81 = (_gern_k_3_9_15_21_27_33_39_45_51_57_63_69_75_81 + k_tiled)) {
        auto _query_A_99 = A.template stage_into_smem_flat<block_x, k_tiled>(_gern_i_1_7_13_19_25_31_37_43_49_55_61_67_73_79_85_91, (_gern_k_3_9_15_21_27_33_39_45_51_57_63_69_75_81 + 0));

        auto _query_B_100 = B.template stage_into_smem_flat<k_tiled, block_y>((_gern_k_3_9_15_21_27_33_39_45_51_57_63_69_75_81 + 0), (_gern_j_2_8_14_20_26_32_38_44_50_56_62_68_74_80_86 + 0));

        int64_t _gern_i_1_7_13_19_25_31_37_43_49_55_61 = (((((threadIdx.y / 32) / 1) % ((((warp_x + (block_x - 0)) - 1) / warp_x) * 32)) * warp_x) + 0);
        int64_t _gern_j_2_8_14_20_26_32_38_44_50_56 = (((((threadIdx.x / 32) / 1) % ((((warp_y + (block_y - 0)) - 1) / warp_y) * 32)) * warp_y) + 0);
        int64_t _gern_i_1_7_13_19_25_31_37_43_49 = (((threadIdx.x % 32) * thread_x) + 0);
        auto _query_A_102 = _query_A_99.template query_2_reg_no_vector<thread_x, k_tiled>((_gern_i_1_7_13_19_25_31_37_43_49 + (_gern_i_1_7_13_19_25_31_37_43_49_55_61 + 0)), 0);

        int64_t _gern_j_2_8_14_20_26_32_38_44 = (((threadIdx.y % 32) * thread_y) + 0);
        auto _query_C_101 = _query_C_98.template query_2_reg_no_vector<thread_x, thread_y>((_gern_i_1_7_13_19_25_31_37_43_49 + (_gern_i_1_7_13_19_25_31_37_43_49_55_61 + 0)), (_gern_j_2_8_14_20_26_32_38_44 + (_gern_j_2_8_14_20_26_32_38_44_50_56 + 0)));

        auto _query_B_103 = _query_B_100.template query_2_reg_no_vector<k_tiled, thread_y>(0, (_gern_j_2_8_14_20_26_32_38_44 + (_gern_j_2_8_14_20_26_32_38_44_50_56 + 0)));

        for (int64_t _gern_k_3_9_15_21 = 0; (_gern_k_3_9_15_21 < k_tiled); _gern_k_3_9_15_21 = (_gern_k_3_9_15_21 + thread_k)) {
            auto _query_A_104 = _query_A_102.template get_view<thread_x, thread_k>(0, (_gern_k_3_9_15_21 + 0));

            auto _query_B_105 = _query_B_103.template get_view<thread_k, thread_y>((_gern_k_3_9_15_21 + 0), 0);

            matrix_multiply<thread_k>(_query_A_104, _query_B_105, _query_C_101);
        }

        _query_C_98.template insert_2_reg_no_vector<thread_x, thread_y>((_gern_i_1_7_13_19_25_31_37_43_49 + (_gern_i_1_7_13_19_25_31_37_43_49_55_61 + 0)), (_gern_j_2_8_14_20_26_32_38_44 + (_gern_j_2_8_14_20_26_32_38_44_50_56 + 0)), _query_C_101);
    }
}

extern "C" {
int main() {
    constexpr int64_t block_x = 32;
    constexpr int64_t block_y = 32;

    constexpr int64_t k_dim = 1024;

    constexpr int64_t k_tiled = 64;
    constexpr int64_t thread_k = 8;
    constexpr int64_t thread_x = 1;
    constexpr int64_t thread_y = 1;
    constexpr int64_t warp_x = 32;
    constexpr int64_t warp_y = 32;

    impl::MatrixGPU<k_dim, k_dim, k_dim, 1> A;
    A.ascending();
    impl::MatrixGPU<k_dim, k_dim, k_dim, 1> B;
    B.ascending();
    impl::MatrixGPU<k_dim, k_dim, k_dim, 1> C;
    C.ascending();

    int64_t smem_size = 32 * 32 * 8 * 4;

    dim3 grid_106 = dim3((1 * (((block_x + (C.row - 0)) - 1) / block_x)), (1 * (((block_y + (C.col - 0)) - 1) / block_y)), 1);
    dim3 block_107 = dim3((1 * ((((warp_y + (block_y - 0)) - 1) / warp_y) * 32)), (1 * ((((warp_x + (block_x - 0)) - 1) / warp_x) * 32)), 1);
    auto function_sp_108 = function_97<block_x, block_y, k_dim, k_tiled, thread_k, thread_x, thread_y, warp_x, warp_y>;
    cudaFuncSetAttribute(function_sp_108, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    for (int i = 0; i < 10; i++) {
        function_sp_108<<<grid_106, block_107, smem_size>>>(A, B, C, smem_size);
    }

    cudaDeviceSynchronize();

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 200; i++) {
        function_sp_108<<<grid_106, block_107, smem_size>>>(A, B, C, smem_size);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> duration = (end - start);
    auto min = (duration.count() / 200) / 1e6;

    long double measure = k_dim * k_dim * k_dim * 2LL * 1e-9;
    // double measure = 6 * row_val * col_val;
    std::cout << measure / min << std::endl;

    return 0;
}
}
