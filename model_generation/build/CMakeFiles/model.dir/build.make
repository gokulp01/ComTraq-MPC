# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

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
CMAKE_COMMAND = /opt/homebrew/Cellar/cmake/3.27.6/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/Cellar/cmake/3.27.6/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/cockroach/Desktop/multi-agent/model_generation

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/cockroach/Desktop/multi-agent/model_generation/build

# Include any dependencies generated for this target.
include CMakeFiles/model.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/model.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/model.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/model.dir/flags.make

CMakeFiles/model.dir/main.cpp.o: CMakeFiles/model.dir/flags.make
CMakeFiles/model.dir/main.cpp.o: /Users/cockroach/Desktop/multi-agent/model_generation/main.cpp
CMakeFiles/model.dir/main.cpp.o: CMakeFiles/model.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/cockroach/Desktop/multi-agent/model_generation/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/model.dir/main.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/model.dir/main.cpp.o -MF CMakeFiles/model.dir/main.cpp.o.d -o CMakeFiles/model.dir/main.cpp.o -c /Users/cockroach/Desktop/multi-agent/model_generation/main.cpp

CMakeFiles/model.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/model.dir/main.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/cockroach/Desktop/multi-agent/model_generation/main.cpp > CMakeFiles/model.dir/main.cpp.i

CMakeFiles/model.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/model.dir/main.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/cockroach/Desktop/multi-agent/model_generation/main.cpp -o CMakeFiles/model.dir/main.cpp.s

CMakeFiles/model.dir/model_new.cpp.o: CMakeFiles/model.dir/flags.make
CMakeFiles/model.dir/model_new.cpp.o: /Users/cockroach/Desktop/multi-agent/model_generation/model_new.cpp
CMakeFiles/model.dir/model_new.cpp.o: CMakeFiles/model.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/cockroach/Desktop/multi-agent/model_generation/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/model.dir/model_new.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/model.dir/model_new.cpp.o -MF CMakeFiles/model.dir/model_new.cpp.o.d -o CMakeFiles/model.dir/model_new.cpp.o -c /Users/cockroach/Desktop/multi-agent/model_generation/model_new.cpp

CMakeFiles/model.dir/model_new.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/model.dir/model_new.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/cockroach/Desktop/multi-agent/model_generation/model_new.cpp > CMakeFiles/model.dir/model_new.cpp.i

CMakeFiles/model.dir/model_new.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/model.dir/model_new.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/cockroach/Desktop/multi-agent/model_generation/model_new.cpp -o CMakeFiles/model.dir/model_new.cpp.s

CMakeFiles/model.dir/config.cpp.o: CMakeFiles/model.dir/flags.make
CMakeFiles/model.dir/config.cpp.o: /Users/cockroach/Desktop/multi-agent/model_generation/config.cpp
CMakeFiles/model.dir/config.cpp.o: CMakeFiles/model.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/cockroach/Desktop/multi-agent/model_generation/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/model.dir/config.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/model.dir/config.cpp.o -MF CMakeFiles/model.dir/config.cpp.o.d -o CMakeFiles/model.dir/config.cpp.o -c /Users/cockroach/Desktop/multi-agent/model_generation/config.cpp

CMakeFiles/model.dir/config.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/model.dir/config.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/cockroach/Desktop/multi-agent/model_generation/config.cpp > CMakeFiles/model.dir/config.cpp.i

CMakeFiles/model.dir/config.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/model.dir/config.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/cockroach/Desktop/multi-agent/model_generation/config.cpp -o CMakeFiles/model.dir/config.cpp.s

CMakeFiles/model.dir/uuv.cpp.o: CMakeFiles/model.dir/flags.make
CMakeFiles/model.dir/uuv.cpp.o: /Users/cockroach/Desktop/multi-agent/model_generation/uuv.cpp
CMakeFiles/model.dir/uuv.cpp.o: CMakeFiles/model.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/cockroach/Desktop/multi-agent/model_generation/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/model.dir/uuv.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/model.dir/uuv.cpp.o -MF CMakeFiles/model.dir/uuv.cpp.o.d -o CMakeFiles/model.dir/uuv.cpp.o -c /Users/cockroach/Desktop/multi-agent/model_generation/uuv.cpp

CMakeFiles/model.dir/uuv.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/model.dir/uuv.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/cockroach/Desktop/multi-agent/model_generation/uuv.cpp > CMakeFiles/model.dir/uuv.cpp.i

CMakeFiles/model.dir/uuv.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/model.dir/uuv.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/cockroach/Desktop/multi-agent/model_generation/uuv.cpp -o CMakeFiles/model.dir/uuv.cpp.s

# Object files for target model
model_OBJECTS = \
"CMakeFiles/model.dir/main.cpp.o" \
"CMakeFiles/model.dir/model_new.cpp.o" \
"CMakeFiles/model.dir/config.cpp.o" \
"CMakeFiles/model.dir/uuv.cpp.o"

# External object files for target model
model_EXTERNAL_OBJECTS =

model: CMakeFiles/model.dir/main.cpp.o
model: CMakeFiles/model.dir/model_new.cpp.o
model: CMakeFiles/model.dir/config.cpp.o
model: CMakeFiles/model.dir/uuv.cpp.o
model: CMakeFiles/model.dir/build.make
model: CMakeFiles/model.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/cockroach/Desktop/multi-agent/model_generation/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable model"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/model.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/model.dir/build: model
.PHONY : CMakeFiles/model.dir/build

CMakeFiles/model.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/model.dir/cmake_clean.cmake
.PHONY : CMakeFiles/model.dir/clean

CMakeFiles/model.dir/depend:
	cd /Users/cockroach/Desktop/multi-agent/model_generation/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/cockroach/Desktop/multi-agent/model_generation /Users/cockroach/Desktop/multi-agent/model_generation /Users/cockroach/Desktop/multi-agent/model_generation/build /Users/cockroach/Desktop/multi-agent/model_generation/build /Users/cockroach/Desktop/multi-agent/model_generation/build/CMakeFiles/model.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/model.dir/depend

