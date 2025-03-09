cmake_minimum_required(VERSION 3.30)
project(Gern_Apps)


macro(add_gern_app name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} Gern::Gern)
endmacro()

macro(add_gern_app_gpu name)
    enable_language(CUDA)
    add_executable(${name} ${name}.cu)
    target_link_libraries(${name} Gern::Gern)
    target_link_libraries(${name} CUDA::cudart)
endmacro()


# include(current_path.h)

# Find gern and the cuda toolkit.
find_package(Gern REQUIRED)
find_package(CUDAToolkit REQUIRED)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -no-pie")

add_gern_app(1_trivial)
add_gern_app(2_jit)
add_gern_app(3_multi_func)
add_gern_app(4_multi_funct_matrix)
add_gern_app(5_tile_program)
add_gern_app(6_more_tilings)
add_gern_app(7_reduction)
add_gern_app(8_matrix_multiply)
add_gern_app_gpu(9_gpu_trivial)
add_gern_app_gpu(10_gpu_tiling)
add_gern_app_gpu(11_gpu_smem)