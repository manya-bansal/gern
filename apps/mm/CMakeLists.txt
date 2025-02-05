cmake_minimum_required(VERSION 3.30)
project(mm)

find_package(Gern REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(mm mm.cpp)
target_link_libraries(mm PRIVATE Gern::Gern)
target_link_libraries(mm PRIVATE CUDA::cudart)
