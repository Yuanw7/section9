#!/bin/bash

# Create directories (handles spaces in names)
mkdir -p "odd" "even"

# Process files
for file in *; do
    # Skip directories and non-files
    if [[ -f "$file" ]]; then
        # Extract the last number in the filename
        if [[ "$file" =~ ([0-9]+)[^0-9]*$ ]]; then
            index="${BASH_REMATCH[1]}"
            
            # Determine odd/even
            if (( index % 2 == 0 )); then
                dest="even"
            else
                dest="odd"
            fi
            
            # Move the file
            mv -- "$file" "$dest/"
        fi
    fi
done

echo "Files organized into odd/even directories!"
