cmake_minimum_required(VERSION 3.30)
project(Gern_Apps)


macro(add_gern_app name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} Gern::Gern)
endmacro()


# include(current_path.h)

# Find gern and the cuda toolkit.
find_package(Gern REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_gern_app(1_trivial)
add_gern_app(2_jit)
add_gern_app(3_multi_func)
add_gern_app(5_tile_program)
add_gern_app(6_more_tilings)