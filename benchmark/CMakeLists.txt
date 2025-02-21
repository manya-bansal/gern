find_package(OpenBLAS REQUIRED)


add_executable(gern_benchmark)

configure_file(
    "config.h.in"
    "config.h"
)

set(LIB_HEADERS
    lib/annot/cpu-matrix.h
    lib/impl/cpu-matrix.h
)


target_sources(
    gern_benchmark
    PRIVATE
        benchmark.cpp
    PRIVATE
    FILE_SET HEADERS
    FILES
       benchmark-utils.h
       ${LIB_HEADERS} 

)

include_directories(${PROJECT_BINARY_DIR}/benchmark)

target_compile_options(gern_benchmark PRIVATE -O3 -march=native)
target_link_libraries(gern_benchmark PRIVATE Gern::Gern OpenBLAS::OpenBLAS)

