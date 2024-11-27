cmake_minimum_required(VERSION 3.18)
project(Gern)
set(CMAKE_CXX_STANDARD 17)

set(GERN_PROJECT_DIR ${CMAKE_CURRENT_LIST_DIR})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(_DEBUG)
endif()

include_directories(include)


################################
# GTest
################################
ADD_SUBDIRECTORY (googletest)
enable_testing()
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

add_executable(test_exec main.cpp)

# Add compiler flags for the test executable
target_compile_options(test_exec PRIVATE
    -Wall
    -Wextra
    -pedantic
    -stdlib=libc++
    -Wc++17-attribute-extensions
    -O3
    -Wl,-no_compact_unwind
    -fvisibility=hidden
)

target_link_libraries(test_exec
    ${CMAKE_DL_LIBS}
)


add_subdirectory(src)
add_subdirectory(test)
