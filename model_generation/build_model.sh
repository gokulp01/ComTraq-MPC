#!/bin/bash

# Define the build directory
BUILD_DIR="build"

# Check if the build directory exists, if not, create it
if [ ! -d "$BUILD_DIR" ]; then
    echo "Creating build directory..."
    mkdir $BUILD_DIR
fi

# Navigate to the build directory
cd $BUILD_DIR

# Generate the build system with CMake
echo "Generating build system..."
cmake ..

# Build the project
echo "Building the project..."
cmake --build .

echo "Model built, copying model executable. run ./model to run the model"
mv model ../


# Navigate back to the original directory
cd ..
