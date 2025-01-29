#!/bin/bash
# This script deletes all .bin files from a specified directory

if [ -z "$1" ]; then
    echo "Usage: ./file_remove.sh <directory>"
    exit 1
fi

TARGET_DIR="$1"

# Delete all .bin files in the target directory
rm -f "$TARGET_DIR"/*.bin
echo "✅ Deleted all .bin files in $TARGET_DIR"
