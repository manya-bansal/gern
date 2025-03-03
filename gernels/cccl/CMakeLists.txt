cmake_minimum_required(VERSION 3.30)
project(cccl)

add_executable(rdom_mimic rdom_mimic.cpp)
target_link_libraries(rdom_mimic PRIVATE Gern::Gern)
target_link_libraries(rdom_mimic PRIVATE CUDA::cudart)

add_executable(global_reduce global_reduce.cpp)
target_link_libraries(global_reduce PRIVATE Gern::Gern)
target_link_libraries(global_reduce PRIVATE CUDA::cudart)
