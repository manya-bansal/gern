cmake_minimum_required(VERSION 3.30)
project(softmax)

find_package(Gern REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(softmax softmax.cpp)
target_link_libraries(softmax PRIVATE Gern::Gern)
target_link_libraries(softmax PRIVATE CUDA::cudart)
