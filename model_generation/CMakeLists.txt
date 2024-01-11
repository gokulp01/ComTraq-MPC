# CMakeLists.txt

cmake_minimum_required(VERSION 3.10)
project(UUVModel)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add all the source files
set(SOURCES
    main.cpp
    model_new.cpp
    config.cpp
    uuv.cpp
)

add_executable(model ${SOURCES})
