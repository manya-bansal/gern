file(GLOB TEST_SOURCES *.cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_executable(gern_test ${TEST_SOURCES})

target_link_libraries(gern_test
gtest_main
${CMAKE_DL_LIBS}
gern
)
