find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(gern_test)

set(LIB_HEADERS
    library/array/annot/cpu-array.h
    library/array/annot/cpu-array-template.h
    library/array/annot/gpu-array.h
    library/array/impl/cpu-array.h
    library/array/impl/cpu-array-template.h
    
    library/matrix/annot/cpu-matrix.h
    library/matrix/annot/gpu-matrix.h
    library/matrix/impl/cpu-matrix.h
     
    library/scalar/annot/cpu-scalar.h
    library/scalar/impl/cpu-scalar.h

)

set(TEST_SOURCES
    test-composable.cpp
    test-cpu-array.cpp
    test-cpu-matrix.cpp
    test-cpu-mm.cpp
    test-cpu-static-store.cpp
    test-cpu-reduction.cpp
    test-functions.cpp
    test-function-ptr.cpp
    test-grid-constraints.cpp
    test-map-grid.cpp
    test-nodes.cpp
    test-pipeline.cpp
    test-set-unit.cpp
    # test-resolve-constraints.cpp
    test-runner.cpp
    test-stage.cpp
)


if (Gern_CUDA_ARCH)
    find_package(CUDAToolkit REQUIRED)
    set_target_properties(gern_test PROPERTIES CUDA_ARCHITECTURES "${Gern_CUDA_ARCH}")
    enable_language(CUDA)

    # Adding all the GPU impl headers.
    list(APPEND LIB_HEADERS
    test-gpu-utils.h
    library/array/impl/gpu-array.h
    library/matrix/impl/gpu-matrix.h
    library/smem_allocator/sh_malloc.h)
    # Adding all the GPU test.
    list(APPEND TEST_SOURCES
    test-gpu-array.cu
    test-set-grid.cu
    test-gpu-matrix.cu
    test-dynamic-shared.cu)
endif()


configure_file(
    "config.h.in"
    "config.h"
)

target_sources(
    gern_test
    PRIVATE
        ${TEST_SOURCES}
    PRIVATE
    FILE_SET HEADERS
    FILES
       test-utils.h
       ${LIB_HEADERS} 
)

include_directories(${PROJECT_BINARY_DIR}/test)
target_link_libraries(gern_test PRIVATE Gern::Gern GTest::gtest_main)
# target_link_libraries(gern_test PRIVATE ginac::ginac)

if (DEFINED Gern_CUDA_ARCH)
    target_link_libraries(gern_test PRIVATE CUDA::cudart -lrt)
endif ()

# This is needed to avoid the "not relocatable" error for extern __shared__ shmem.
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -no-pie")

gtest_discover_tests(gern_test)
