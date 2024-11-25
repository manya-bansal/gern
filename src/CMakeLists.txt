file(GLOB SRC_FILES *.cpp)

set(GERN_SRC_DIRS .)

foreach(dir ${GERN_SRC_DIRS})
  file(GLOB SRC_FILES ${SRC_FILES} ${dir}/*.cpp)
endforeach()

# include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(../include)
add_library(gern STATIC ${SRC_FILES})