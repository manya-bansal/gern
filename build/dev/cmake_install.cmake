# Install script for directory: /home/manya/gern

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/manya/gern_install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "RelWithDebInfo")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/manya/gern/build/dev/test/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Gern_Development" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/manya/gern/build/dev/libGern.a")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Gern_Development" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/annotations" TYPE FILE FILES
    "/home/manya/gern/include/gern/annotations/abstract_nodes.h"
    "/home/manya/gern/include/gern/annotations/abstract_function.h"
    "/home/manya/gern/include/gern/annotations/argument.h"
    "/home/manya/gern/include/gern/annotations/argument_visitor.h"
    "/home/manya/gern/include/gern/annotations/data_dependency_language.h"
    "/home/manya/gern/include/gern/annotations/datatypes.h"
    "/home/manya/gern/include/gern/annotations/expr.h"
    "/home/manya/gern/include/gern/annotations/expr_nodes.h"
    "/home/manya/gern/include/gern/annotations/grid.h"
    "/home/manya/gern/include/gern/annotations/lang_nodes.h"
    "/home/manya/gern/include/gern/annotations/rewriter_helpers.h"
    "/home/manya/gern/include/gern/annotations/std_less_specialization.h"
    "/home/manya/gern/include/gern/annotations/stmt.h"
    "/home/manya/gern/include/gern/annotations/stmt_nodes.h"
    "/home/manya/gern/include/gern/annotations/visitor.h"
    "/home/manya/gern/include/gern/annotations/rewriter.h"
    "/home/manya/gern/include/gern/annotations/shared_memory_manager.h"
    )
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/codegen" TYPE FILE FILES
    "/home/manya/gern/include/gern/codegen/codegen.h"
    "/home/manya/gern/include/gern/codegen/codegen_ir.h"
    "/home/manya/gern/include/gern/codegen/codegen_visitor.h"
    "/home/manya/gern/include/gern/codegen/codegen_printer.h"
    "/home/manya/gern/include/gern/codegen/lower.h"
    "/home/manya/gern/include/gern/codegen/lower_visitor.h"
    "/home/manya/gern/include/gern/codegen/finalizer.h"
    )
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/compose" TYPE FILE FILES
    "/home/manya/gern/include/gern/compose/compose.h"
    "/home/manya/gern/include/gern/compose/composable.h"
    "/home/manya/gern/include/gern/compose/composable_visitor.h"
    "/home/manya/gern/include/gern/compose/composable_node.h"
    "/home/manya/gern/include/gern/compose/runner.h"
    )
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/utils" TYPE FILE FILES
    "/home/manya/gern/include/gern/utils/debug.h"
    "/home/manya/gern/include/gern/utils/error.h"
    "/home/manya/gern/include/gern/utils/name_generator.h"
    "/home/manya/gern/include/gern/utils/scoped_map.h"
    "/home/manya/gern/include/gern/utils/printer.h"
    "/home/manya/gern/include/gern/utils/uncopyable.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Gern_Development" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/gern" TYPE FILE FILES
    "/home/manya/gern/build/dev/cmake/GernConfig.cmake"
    "/home/manya/gern/build/dev/cmake/GernConfigVersion.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Gern_Development" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/gern/gern-targets.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/gern/gern-targets.cmake"
         "/home/manya/gern/build/dev/CMakeFiles/Export/3be9115fdff0bad55211c8b96331162e/gern-targets.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/gern/gern-targets-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/gern/gern-targets.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/gern" TYPE FILE FILES "/home/manya/gern/build/dev/CMakeFiles/Export/3be9115fdff0bad55211c8b96331162e/gern-targets.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/gern" TYPE FILE FILES "/home/manya/gern/build/dev/CMakeFiles/Export/3be9115fdff0bad55211c8b96331162e/gern-targets-relwithdebinfo.cmake")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_COMPONENT MATCHES "^[a-zA-Z0-9_.+-]+$")
    set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
  else()
    string(MD5 CMAKE_INST_COMP_HASH "${CMAKE_INSTALL_COMPONENT}")
    set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INST_COMP_HASH}.txt")
    unset(CMAKE_INST_COMP_HASH)
  endif()
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
  file(WRITE "/home/manya/gern/build/dev/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
