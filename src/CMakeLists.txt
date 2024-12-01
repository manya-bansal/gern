file(GLOB SRC_FILES *.cpp)

set(GERN_SRC_DIRS . annotations utils resolve_constraints)

foreach(dir ${GERN_SRC_DIRS})
  file(GLOB SRC_FILES ${SRC_FILES} ${dir}/*.cpp)
endforeach()

include_directories(../include)
add_library(gern STATIC ${SRC_FILES})
target_link_libraries(gern PUBLIC ${SYMENGINE_LIBRARIES} ginac::ginac)