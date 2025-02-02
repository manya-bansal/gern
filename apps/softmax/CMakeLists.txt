cmake_minimum_required(VERSION 3.30)
project(softmax)

find_package(Gern REQUIRED)

add_executable(softmax main.cpp)
target_link_libraries(softmax PRIVATE Gern::Gern)
