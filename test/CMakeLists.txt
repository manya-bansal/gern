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
        functions/elementwise.h
        functions/reduction.h
        library/array/cpu-array-lib.h
        library/array/gpu-array-lib.h
)

include_directories(${PROJECT_BINARY_DIR}/test)
target_link_libraries(gern_test PRIVATE Gern::Gern GTest::gtest_main)

gtest_discover_tests(gern_test)
