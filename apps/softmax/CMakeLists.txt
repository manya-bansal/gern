cmake_minimum_required(VERSION 3.30)
project(softmax)

find_package(Gern REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(softmax)


target_sources(
    softmax
    PRIVATE
        main.cpp 
    PRIVATE
    FILE_SET HEADERS
    FILES
        annot/adt.h 
        annot/functions.h 
        gpu-matrix-const.h
)

target_link_libraries(softmax PRIVATE Gern::Gern)
target_link_libraries(softmax PRIVATE CUDA::cudart)
