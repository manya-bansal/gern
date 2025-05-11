add_executable(mm_benchmark)

configure_file(
    "config.h.in"
    "config.h"
)

set(LIB_HEADERS
    lib/annot/cpu-matrix.h
    lib/impl/cpu-matrix.h
)

target_sources(
    mm_benchmark
    PRIVATE
        benchmark.cpp
    PRIVATE
    FILE_SET HEADERS
    FILES
       benchmark-utils.h
       ${LIB_HEADERS} 
)

# Add the directory where config.h is generated to the include directories
target_include_directories(mm_benchmark PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_compile_options(mm_benchmark PRIVATE -O3 -march=native)

find_package(OpenBLAS REQUIRED)
target_link_libraries(mm_benchmark PRIVATE Gern::Gern ${OpenBLAS_LIBRARIES})
