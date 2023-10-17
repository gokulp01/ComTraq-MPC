#!/bin/bash

# Change to the build directory
cd build

# Build the project
cmake --build .

# Change back to the parent directory
cd ..

# Remove the existing shared object, if it exists
rm -f model.cpython-310-x86_64-linux-gnu.so

# Copy the new shared object from the build directory
cp build/model.cpython-310-x86_64-linux-gnu.so .

# List the contents of the directory
ls
