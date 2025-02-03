cmake_minimum_required(VERSION 3.30)
project(blur)

find_package(Gern REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(blur blur.cpp)
target_link_libraries(blur PRIVATE Gern::Gern)
target_link_libraries(blur PRIVATE CUDA::cudart)
