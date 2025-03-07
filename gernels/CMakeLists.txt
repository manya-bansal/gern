cmake_minimum_required(VERSION 3.30)
project(Gern_Apps)


macro(add_gernel name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE Gern::Gern CUDA::cudart)
endmacro()


include(current_path.h)

# Find gern and the cuda toolkit.
find_package(Gern REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(CCCL)

add_subdirectory(cccl)
add_subdirectory(mm)