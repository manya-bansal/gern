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
    test-nodes.cpp
    test-pipeline.cpp
    test-resolve-constraints.cpp
    test-runner.cpp
    test-functions.cpp
    test-grid-property.cpp
    test-cpu-array.cpp
    test-cpu-matrix.cpp
    test-cpu-static-store.cpp
)


if (Gern_CUDA_ARCH)
    find_package(CUDAToolkit REQUIRED)
    # Adding all the GPU impl headers.
    list(APPEND LIB_HEADERS
    test-gpu-utils.h
    library/array/impl/gpu-array.h
    library/matrix/impl/gpu-matrix.h)
    # Adding all the GPU test.
    list(APPEND TEST_SOURCES
    test-gpu-array.cpp
    test-gpu-matrix.cpp)
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
target_link_libraries(gern_test PRIVATE ginac::ginac)

if (DEFINED Gern_CUDA_ARCH)
    target_link_libraries(gern_test PRIVATE CUDA::cudart)
endif ()

gtest_discover_tests(gern_test)
