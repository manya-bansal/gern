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
        test-lowering-cpu.cpp
        test-lowering-gpu.cpp
    PRIVATE
    FILE_SET HEADERS
    FILES
        test-utils.h

        library/array/annot/cpu-array.h
        library/array/annot/gpu-array.h
        library/array/impl/cpu-array.h
        library/array/impl/gpu-array.h
)

include_directories(${PROJECT_BINARY_DIR}/test)
target_link_libraries(gern_test PRIVATE Gern::Gern GTest::gtest_main)

gtest_discover_tests(gern_test)
