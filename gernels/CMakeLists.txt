cmake_minimum_required(VERSION 3.30)
project(Gern_Apps)
enable_language(CUDA)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -no-pie")

macro(add_gernel name)
    add_executable(${name} ${name}.cu)
    target_link_libraries(${name} PRIVATE Gern::Gern CUDA::cudart)
    set_target_properties(${name} PROPERTIES CUDA_ARCHITECTURES "89")
endmacro()


include(current_path.h)

# Find gern and the cuda toolkit.
find_package(Gern REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(CCCL)

add_subdirectory(cccl)
add_subdirectory(mm)