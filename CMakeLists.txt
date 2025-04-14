cmake_minimum_required(VERSION 3.30)
project(Gern VERSION 0.1.0)

if (PROJECT_IS_TOP_LEVEL)
    include(CTest)
endif ()

##
# Find dependencies
# find_package(ginac 1.8.7 REQUIRED)


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
        src/annotations/abstract_function.cpp
        src/annotations/argument.cpp
        src/annotations/argument_visitor.cpp
        src/annotations/data_dependency_language.cpp
        src/annotations/datatypes.cpp
        src/annotations/grid.cpp
        src/annotations/lang_nodes.cpp
        src/annotations/visitor.cpp
        src/compose/compose.cpp
        src/compose/composable.cpp
        src/compose/composable_visitor.cpp
        src/codegen/concretize.cpp
        src/codegen/codegen.cpp
        src/codegen/codegen_ir.cpp
        src/codegen/codegen_printer.cpp
        src/codegen/lower.cpp
        src/codegen/lower_visitor.cpp
        src/codegen/runner.cpp
        src/codegen/finalizer.cpp
        # src/resolve_constraints/resolve_constraints.cpp
        src/utils/name_generator.cpp
    PUBLIC
    FILE_SET HEADERS
    BASE_DIRS 
        include/gern/
    FILES
        include/gern/annotations/abstract_nodes.h
        include/gern/annotations/abstract_function.h
        include/gern/annotations/argument.h
        include/gern/annotations/argument_visitor.h
        include/gern/annotations/data_dependency_language.h
        include/gern/annotations/datatypes.h
        include/gern/annotations/expr.h
        include/gern/annotations/expr_nodes.h
        include/gern/annotations/grid.h
        include/gern/annotations/lang_nodes.h
        include/gern/annotations/rewriter_helpers.h
        include/gern/annotations/std_less_specialization.h
        include/gern/annotations/stmt.h
        include/gern/annotations/stmt_nodes.h
        include/gern/annotations/visitor.h
        include/gern/annotations/rewriter.h
        include/gern/annotations/shared_memory_manager.h
        # includgern/e/resolve_constraints/resolve_constraints.h
        include/gern/compose/compose.h
        include/gern/compose/composable.h
        include/gern/compose/composable_visitor.h
        include/gern/compose/composable_node.h
        include/gern/compose/runner.h
        include/gern/codegen/helpers/assert_device_properties.h
<<<<<<< HEAD
=======
        include/gern/codegen/helpers/check_last_error.h
>>>>>>> 353d78bfdaab9e4c9a0ecf59ca989e0fe4f86ba2
        include/gern/codegen/concretize.h
        include/gern/codegen/codegen.h
        include/gern/codegen/codegen_ir.h
        include/gern/codegen/codegen_visitor.h
        include/gern/codegen/codegen_printer.h
        include/gern/codegen/lower.h
        include/gern/codegen/lower_visitor.h
        include/gern/codegen/finalizer.h
        include/gern/utils/debug.h
        include/gern/utils/error.h
        include/gern/utils/name_generator.h
        include/gern/utils/scoped_map.h
        include/gern/utils/printer.h
        include/gern/utils/uncopyable.h
)

target_compile_definitions(Gern_Gern PUBLIC $<$<CONFIG:Debug>:GERN_DEBUG_BUILD>)
target_compile_features(Gern_Gern PUBLIC cxx_std_20)
target_link_libraries(Gern_Gern PRIVATE ${CMAKE_DL_LIBS})

##
# Testing

option(Gern_BUILD_TESTING "Build Gern's test suite" "${BUILD_TESTING}")
if (Gern_BUILD_TESTING)
    add_subdirectory(test)
endif ()

if (Gern_CUDA_ARCH)
    message("Running GPU testers in this build.")
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

