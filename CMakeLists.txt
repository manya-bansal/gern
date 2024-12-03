cmake_minimum_required(VERSION 3.30)
project(Gern VERSION 0.1.0)

if (PROJECT_IS_TOP_LEVEL)
    include(CTest)
endif ()

##
# Find dependencies

find_package(ginac 1.8.7 REQUIRED)

##
# Main library target

add_library(Gern_Gern STATIC)
add_library(Gern::Gern ALIAS Gern_Gern)
set_target_properties(
    Gern_Gern
    PROPERTIES
    EXPORT_NAME Gern
    OUTPUT_NAME Gern
)

target_sources(
    Gern_Gern
    PRIVATE
        src/annotations/abstract_data.cpp
        src/annotations/abstract_function.cpp
        src/annotations/data_dependency_language.cpp
        src/annotations/datatypes.cpp
        src/annotations/lang_nodes.cpp
        src/annotations/visitor.cpp
        src/compose/compose.cpp
        src/resolve_constraints/resolve_constraints.cpp
        src/utils/name_generator.cpp
    PUBLIC
    FILE_SET HEADERS
    BASE_DIRS include
    FILES
        include/annotations/abstract_nodes.h
        include/annotations/abstract_function.h
        include/annotations/arguments.h
        include/annotations/data_dependency_language.h
        include/annotations/datatypes.h
        include/annotations/lang_nodes.h
        include/annotations/visitor.h
        include/resolve_constraints/resolve_constraints.h
        include/compose/compose.h
        include/utils/debug.h
        include/utils/error.h
        include/utils/name_generator.h
        include/utils/uncopyable.h
)
target_compile_definitions(Gern_Gern PUBLIC $<$<CONFIG:Debug>:GERN_DEBUG_BUILD>)
target_compile_features(Gern_Gern PUBLIC cxx_std_17)
target_link_libraries(Gern_Gern PRIVATE ginac::ginac)

##
# Testing

option(Gern_BUILD_TESTING "Build Gern's test suite" "${BUILD_TESTING}")
if (Gern_BUILD_TESTING)
    add_subdirectory(test)
endif ()

##
# Installation and packaging

option(Gern_INSTALL_RULES "Include Gern's install rules" "${PROJECT_IS_TOP_LEVEL}")
if (Gern_INSTALL_RULES)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    # Install executable and library targets
    install(
        TARGETS Gern_Gern
        EXPORT gern-targets
        RUNTIME COMPONENT Gern_Runtime
        LIBRARY COMPONENT Gern_Runtime
        NAMELINK_COMPONENT Gern_Development
        ARCHIVE COMPONENT Gern_Development
        FILE_SET HEADERS COMPONENT Gern_Development
    )

    # Install CMake package files
    set(Gern_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/gern"
        CACHE STRING "Install destination for Gern's CMake package")

    configure_package_config_file(
        cmake/GernConfig.cmake.in cmake/GernConfig.cmake
        INSTALL_DESTINATION "${Gern_INSTALL_CMAKEDIR}"
        NO_SET_AND_CHECK_MACRO
    )

    write_basic_package_version_file(
        cmake/GernConfigVersion.cmake COMPATIBILITY SameMajorVersion
    )

    install(
        FILES
        "${CMAKE_CURRENT_BINARY_DIR}/cmake/GernConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/cmake/GernConfigVersion.cmake"
        DESTINATION "${Gern_INSTALL_CMAKEDIR}"
        COMPONENT Gern_Development
    )

    install(
        EXPORT gern-targets
        DESTINATION "${Gern_INSTALL_CMAKEDIR}"
        NAMESPACE Gern::
        FILE gern-targets.cmake
        COMPONENT Gern_Development
    )
endif ()

