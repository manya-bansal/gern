cmake_minimum_required(VERSION 3.30)
project(mm)

find_package(Gern REQUIRED)
find_package(CUDAToolkit REQUIRED)


add_gernel(mm_trivial)
add_gernel(mm_registers)
add_gernel(mm_shared)


# Step by Step kernels
add_gernel(kernel_1)
add_gernel(kernel_2)
add_gernel(kernel_3)