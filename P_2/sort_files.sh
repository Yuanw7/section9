#!/bin/bash
# This script separates odd and even indexed binary files

DATA_DIR="/root/Desktop/yambo/P_2/electron_scattering_data"
ODD_DIR="$DATA_DIR/odd"
EVEN_DIR="$DATA_DIR/even"

# Ensure directories exist
mkdir -p "$ODD_DIR" "$EVEN_DIR"

# Loop through each file
for file in "$DATA_DIR"/*.bin; do
    # Extract the index number using regex
    filename=$(basename "$file")
    index=$(echo "$filename" | grep -oP '\d+(?=\.bin)')
    
    # Check if index is odd or even
    if (( index % 2 == 0 )); then
        mv "$file" "$EVEN_DIR/"
    else
        mv "$file" "$ODD_DIR/"
    fi
done

echo "✅ Files sorted into '$ODD_DIR' and '$EVEN_DIR'"
