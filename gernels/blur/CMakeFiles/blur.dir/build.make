# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.30

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/manya/.local/lib/python3.10/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/manya/.local/lib/python3.10/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/manya/gern/apps

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/manya/gern/apps/build

# Include any dependencies generated for this target.
include blur/CMakeFiles/blur.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include blur/CMakeFiles/blur.dir/compiler_depend.make

# Include the progress variables for this target.
include blur/CMakeFiles/blur.dir/progress.make

# Include the compile flags for this target's objects.
include blur/CMakeFiles/blur.dir/flags.make

blur/CMakeFiles/blur.dir/blur.cpp.o: blur/CMakeFiles/blur.dir/flags.make
blur/CMakeFiles/blur.dir/blur.cpp.o: /home/manya/gern/apps/blur/blur.cpp
blur/CMakeFiles/blur.dir/blur.cpp.o: blur/CMakeFiles/blur.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/manya/gern/apps/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object blur/CMakeFiles/blur.dir/blur.cpp.o"
	cd /home/manya/gern/apps/build/blur && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT blur/CMakeFiles/blur.dir/blur.cpp.o -MF CMakeFiles/blur.dir/blur.cpp.o.d -o CMakeFiles/blur.dir/blur.cpp.o -c /home/manya/gern/apps/blur/blur.cpp

blur/CMakeFiles/blur.dir/blur.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/blur.dir/blur.cpp.i"
	cd /home/manya/gern/apps/build/blur && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/manya/gern/apps/blur/blur.cpp > CMakeFiles/blur.dir/blur.cpp.i

blur/CMakeFiles/blur.dir/blur.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/blur.dir/blur.cpp.s"
	cd /home/manya/gern/apps/build/blur && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/manya/gern/apps/blur/blur.cpp -o CMakeFiles/blur.dir/blur.cpp.s

# Object files for target blur
blur_OBJECTS = \
"CMakeFiles/blur.dir/blur.cpp.o"

# External object files for target blur
blur_EXTERNAL_OBJECTS =

blur/blur: blur/CMakeFiles/blur.dir/blur.cpp.o
blur/blur: blur/CMakeFiles/blur.dir/build.make
blur/blur: /home/manya/gern_install/lib/libGern.a
blur/blur: /usr/local/cuda-12.3/targets/x86_64-linux/lib/libcudart.so
blur/blur: /usr/lib/x86_64-linux-gnu/librt.so
blur/blur: blur/CMakeFiles/blur.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/manya/gern/apps/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable blur"
	cd /home/manya/gern/apps/build/blur && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/blur.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
blur/CMakeFiles/blur.dir/build: blur/blur
.PHONY : blur/CMakeFiles/blur.dir/build

blur/CMakeFiles/blur.dir/clean:
	cd /home/manya/gern/apps/build/blur && $(CMAKE_COMMAND) -P CMakeFiles/blur.dir/cmake_clean.cmake
.PHONY : blur/CMakeFiles/blur.dir/clean

blur/CMakeFiles/blur.dir/depend:
	cd /home/manya/gern/apps/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/manya/gern/apps /home/manya/gern/apps/blur /home/manya/gern/apps/build /home/manya/gern/apps/build/blur /home/manya/gern/apps/build/blur/CMakeFiles/blur.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : blur/CMakeFiles/blur.dir/depend

