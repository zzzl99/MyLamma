# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/lz/MyLLama

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lz/MyLLama/cmake-build-debug

# Include any dependencies generated for this target.
include test/CMakeFiles/test_llm.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include test/CMakeFiles/test_llm.dir/compiler_depend.make

# Include the progress variables for this target.
include test/CMakeFiles/test_llm.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/test_llm.dir/flags.make

test/CMakeFiles/test_llm.dir/test_tensor.cpp.o: test/CMakeFiles/test_llm.dir/flags.make
test/CMakeFiles/test_llm.dir/test_tensor.cpp.o: ../test/test_tensor.cpp
test/CMakeFiles/test_llm.dir/test_tensor.cpp.o: test/CMakeFiles/test_llm.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lz/MyLLama/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/CMakeFiles/test_llm.dir/test_tensor.cpp.o"
	cd /home/lz/MyLLama/cmake-build-debug/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/CMakeFiles/test_llm.dir/test_tensor.cpp.o -MF CMakeFiles/test_llm.dir/test_tensor.cpp.o.d -o CMakeFiles/test_llm.dir/test_tensor.cpp.o -c /home/lz/MyLLama/test/test_tensor.cpp

test/CMakeFiles/test_llm.dir/test_tensor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_llm.dir/test_tensor.cpp.i"
	cd /home/lz/MyLLama/cmake-build-debug/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lz/MyLLama/test/test_tensor.cpp > CMakeFiles/test_llm.dir/test_tensor.cpp.i

test/CMakeFiles/test_llm.dir/test_tensor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_llm.dir/test_tensor.cpp.s"
	cd /home/lz/MyLLama/cmake-build-debug/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lz/MyLLama/test/test_tensor.cpp -o CMakeFiles/test_llm.dir/test_tensor.cpp.s

# Object files for target test_llm
test_llm_OBJECTS = \
"CMakeFiles/test_llm.dir/test_tensor.cpp.o"

# External object files for target test_llm
test_llm_EXTERNAL_OBJECTS =

test/test_llm: test/CMakeFiles/test_llm.dir/test_tensor.cpp.o
test/test_llm: test/CMakeFiles/test_llm.dir/build.make
test/test_llm: /usr/local/lib/libgtest.a
test/test_llm: ../lib/libllama.so
test/test_llm: /usr/local/lib/libglog.so.0.8.0
test/test_llm: /usr/lib/x86_64-linux-gnu/libarmadillo.so
test/test_llm: test/CMakeFiles/test_llm.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lz/MyLLama/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test_llm"
	cd /home/lz/MyLLama/cmake-build-debug/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_llm.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/test_llm.dir/build: test/test_llm
.PHONY : test/CMakeFiles/test_llm.dir/build

test/CMakeFiles/test_llm.dir/clean:
	cd /home/lz/MyLLama/cmake-build-debug/test && $(CMAKE_COMMAND) -P CMakeFiles/test_llm.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/test_llm.dir/clean

test/CMakeFiles/test_llm.dir/depend:
	cd /home/lz/MyLLama/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lz/MyLLama /home/lz/MyLLama/test /home/lz/MyLLama/cmake-build-debug /home/lz/MyLLama/cmake-build-debug/test /home/lz/MyLLama/cmake-build-debug/test/CMakeFiles/test_llm.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/test_llm.dir/depend

