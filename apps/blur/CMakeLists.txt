cmake_minimum_required(VERSION 3.30)
project(blur)

find_package(Gern REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(blur blur.cpp)
add_executable(blur_bench blur_bench.cpp)

target_link_libraries(blur PRIVATE Gern::Gern)
target_link_libraries(blur PRIVATE CUDA::cudart)

target_link_libraries(blur_bench PRIVATE Gern::Gern)
target_link_libraries(blur_bench PRIVATE CUDA::cudart)
