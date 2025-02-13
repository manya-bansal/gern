cmake_minimum_required(VERSION 3.30)
project(cccl)


add_executable(hello_cccl hello_cccl.cpp)
target_link_libraries(hello_cccl PRIVATE Gern::Gern)
target_link_libraries(hello_cccl PRIVATE CUDA::cudart)


add_executable(rdom_mimic rdom_mimic.cpp)
target_link_libraries(rdom_mimic PRIVATE Gern::Gern)
target_link_libraries(rdom_mimic PRIVATE CUDA::cudart)
