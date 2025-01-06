find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(gern_test)

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

        library/array/annot/cpu-array.h
        library/array/annot/gpu-array.h
        library/array/impl/cpu-array.h
        library/array/impl/gpu-array.h
        library/matrix/annot/cpu-matrix.h
        library/matrix/annot/gpu-matrix.h
        library/matrix/impl/cpu-matrix.h
        library/matrix/impl/gpu-matrix.h
)

include_directories(${PROJECT_BINARY_DIR}/test)
target_link_libraries(gern_test PRIVATE Gern::Gern GTest::gtest_main)

gtest_discover_tests(gern_test)
