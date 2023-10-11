#!/bin/bash

compile_service () {
  # $1 declaration file
  # $2 output dir
  protoc $1 --python_out=$2
}

compile_model () {
  # $1 declaration file
  # $2 output dir
  protoc --python_out=$2 $1
}

compile_proto() {
  # $1 declaration file
  # $2 output dir
  python -m grpc_tools.protoc -I. --python_out=$2 --grpc_python_out=$2 $1
}

# Check that the needed folders exists.
SERVICE_DIR="Declarations/Service"
MODEL_DIR="Declarations/Model"

if [ ! -d "$SERVICE_DIR" ]; then
  echo "Cannot find the folder ${SERVICE_DIR}. Make sure you are executing the script using as current directory the one where the generate.sh file is located."
  exit -1
fi

if [ ! -d "$MODEL_DIR" ]; then
  echo "Cannot find the folder ${MODEL_DIR}. Make sure you are executing the script using as current directory the one where the generate.sh file is located."
  exit -2
fi

# Check if output folder exists.
OUTPUT_DIR="Generated"
if [ -d "$OUTPUT_DIR" ]; then
  echo "Generated folder already present, deleting it."
  rm -rf $OUTPUT_DIR
fi
mkdir $OUTPUT_DIR
touch $OUTPUT_DIR/__init__.py  # Add __init__.py to the Generated directory

# List the protobuf files.
SERVICE_FILES=$(find $SERVICE_DIR -type f | grep ".proto")
MODEL_FILES=$(find $MODEL_DIR -type f | grep ".proto")

echo "Compiling services"
for SERVICE_FILE in $SERVICE_FILES; do
  compile_proto $SERVICE_FILE $OUTPUT_DIR
  # Extract directory path from the file path and add __init__.py
  DIR_PATH=$(dirname "$SERVICE_FILE")
  touch "$OUTPUT_DIR/$DIR_PATH/__init__.py"
done

echo "Compiling models"
for MODEL_FILE in $MODEL_FILES; do
  compile_model $MODEL_FILE $OUTPUT_DIR
  # Extract directory path from the file path and add __init__.py
  DIR_PATH=$(dirname "$MODEL_FILE")
  touch "$OUTPUT_DIR/$DIR_PATH/__init__.py"
done
