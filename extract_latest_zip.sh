#!/bin/bash

# Directory containing the zip files
ZIP_DIR="/path/to/zip/directory"

# Get the latest zip file in the directory
LATEST_ZIP=$(ls -t "$ZIP_DIR"/python_files_*.zip 2>/dev/null | head -n 1)

# Check if a zip file is found
if [ -z "$LATEST_ZIP" ]; then
  echo "No zip files found in $ZIP_DIR"
  exit 1
fi

# Extract the latest zip file into the current directory, replacing existing files
unzip -o "$LATEST_ZIP" -d .

echo "Extracted $LATEST_ZIP to the current directory"
