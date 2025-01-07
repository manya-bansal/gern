find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(gern_test)

set(LIB_HEADERS
    library/array/annot/cpu-array.h
    library/array/annot/gpu-array.h
    library/array/impl/cpu-array.h
    
    library/matrix/annot/cpu-matrix.h
    library/matrix/annot/gpu-matrix.h
    library/matrix/impl/cpu-matrix.h
   
)

if(DEFINED Gern_CUDA_ARCH)
    find_package(CUDAToolkit REQUIRED)
    target_compile_definitions(gern_test PUBLIC GERN_GPU_RUNNER)
    # Adding all the GPU impl headers.
    list(APPEND LIB_HEADERS
    library/array/impl/gpu-array.h
    library/matrix/impl/gpu-matrix.h
)
endif()


configure_file(
    "config.h.in"
    "config.h"
)

target_sources(
    gern_test
    PRIVATE
        test-nodes.cpp
        test-resolve-constraints.cpp
        test-functions.cpp
        test-grid-property.cpp
        # Tests for libraries and lowering.
        test-cpu-array.cpp
        test-cpu-matrix.cpp
        test-gpu-array.cpp
        test-gpu-matrix.cpp
    PRIVATE
    FILE_SET HEADERS
    FILES
       test-utils.h
       ${LIB_HEADERS} 
)

include_directories(${PROJECT_BINARY_DIR}/test)
target_link_libraries(gern_test PRIVATE Gern::Gern GTest::gtest_main)
target_link_libraries(gern_test PRIVATE ginac::ginac)

if (DEFINED Gern_CUDA_ARCH)
    target_link_libraries(gern_test PRIVATE CUDA::cudart)
endif ()

gtest_discover_tests(gern_test)
