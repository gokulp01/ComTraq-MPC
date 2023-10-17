cmake_minimum_required(VERSION 3.12...3.20) # Choose your CMake version

project(MyUUVProject VERSION 1.0 DESCRIPTION "UUV project with pybind11" LANGUAGES CXX)

# Add the path to the pybind11 cmake module to the list of paths
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/pybind11/tools") # If you've added pybind11 as a submodule/subdirectory

find_package(pybind11 REQUIRED)  # Look for pybind11, fail if not found

# Define our library named 'model'
pybind11_add_module(model SHARED model.cpp)

# Specify the C++ standard
set_target_properties(model PROPERTIES CXX_STANDARD 17)

# Link against libraries if necessary
# target_link_libraries(model PRIVATE ...)  # If you have any other libraries to link, do so here

# Include directories if necessary
# target_include_directories(model PRIVATE ...) # If you have any include directories, add them here

