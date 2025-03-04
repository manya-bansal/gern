cmake_minimum_required(VERSION 3.30)
project(mm)

find_package(Gern REQUIRED)
find_package(CUDAToolkit REQUIRED)


add_executable(mm_trivial mm_trivial.cpp)
add_executable(mm_registers mm_registers.cpp)

target_link_libraries(mm_trivial PRIVATE Gern::Gern)
target_link_libraries(mm_trivial PRIVATE CUDA::cudart)
target_link_libraries(mm_registers PRIVATE Gern::Gern)
target_link_libraries(mm_registers PRIVATE CUDA::cudart)