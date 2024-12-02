find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(gern_test)
target_sources(
    gern_test
    PRIVATE
        test-nodes.cpp
        test-resolve-constraints.cpp
    PRIVATE
    FILE_SET HEADERS
    FILES
        test-utils.h
)
target_link_libraries(gern_test PRIVATE Gern::Gern GTest::gtest_main)

gtest_discover_tests(gern_test)
