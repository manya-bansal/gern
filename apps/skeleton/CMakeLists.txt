cmake_minimum_required(VERSION 3.30)
project(skeleton)

find_package(Gern REQUIRED)

add_executable(skeleton main.cpp)
target_link_libraries(skeleton PRIVATE Gern::Gern)
