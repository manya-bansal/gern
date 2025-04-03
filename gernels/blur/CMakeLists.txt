cmake_minimum_required(VERSION 3.30)
project(blur)

find_package(Gern REQUIRED)
find_package(CUDAToolkit REQUIRED)


add_gernel(blur)
