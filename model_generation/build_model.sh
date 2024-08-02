#!/bin/bash

BUILD_DIR="build"

if [ ! -d "$BUILD_DIR" ]; then
    echo "Creating build directory..."
    mkdir $BUILD_DIR
fi

cd $BUILD_DIR

echo "Generating build system..."
cmake ..

echo "Building the project..."
cmake --build .

echo "Model built, copying model executable. run ./model to run the model"
mv model ../


cd ..
