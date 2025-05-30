#!/bin/bash

# Check if path argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <directory_path>"
    exit 1
fi

# Store the path argument
path="$1"

# Check if the directory exists
if [ ! -d "$path" ]; then
    echo "Error: Directory $path does not exist"
    exit 1
fi

# Create necessary folders if they don't exist
mkdir -p "$path/unknown40"
mkdir -p "$path/unknown60"
mkdir -p "$path/unknown80"
mkdir -p "$path/basic"

# Process all JSON files in the given path
for file in "$path"/*.json; do
    # Check if file exists (in case no .json files are found)
    [ -f "$file" ] || continue
    
    # Get just the filename
    filename=$(basename "$file")
    
    # Move files based on their names
    if [[ "$filename" == *"unknown40"* ]]; then
        mv "$file" "$path/unknown40/"
    elif [[ "$filename" == *"unknown60"* ]]; then
        mv "$file" "$path/unknown60/"
    elif [[ "$filename" == *"unknown80"* ]]; then
        mv "$file" "$path/unknown80/"
    else
        mv "$file" "$path/basic/"
    fi
done

echo "Files have been organized successfully!"
