#!/bin/bash

# Directory containing the Python files
SOURCE_DIR="./"

# Destination directory for the zip file
DEST_DIR="/home/casa1/owl"

# Get current date and time
CURRENT_DATE_TIME=$(date +"%Y-%m-%d_%H-%M-%S")

# Create zip file name
ZIP_FILE_NAME="python_files_$CURRENT_DATE_TIME.zip"

# Create zip file with all Python files
zip -r "$ZIP_FILE_NAME" "$SOURCE_DIR"/*.py

# Move the zip file to the destination directory
mv "$ZIP_FILE_NAME" "$DEST_DIR"

echo "Python files compressed and moved to $DEST_DIR/$ZIP_FILE_NAME"
