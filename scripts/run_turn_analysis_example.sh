#!/bin/bash

# Script to analyze all subdirectories in a given top-level directory
# Creates corresponding analysis folders for each subdirectory

# Base directories
DEFAULT_TOP_DIR="./example_data/storm_json_final"
TOP_DIR="${1:-$DEFAULT_TOP_DIR}"
PYTHON_SCRIPT="./src/analysis/turn_pairs_analysis.py"

# Check if the input directory exists
if [ ! -d "$TOP_DIR" ]; then
    echo "Error: Directory '$TOP_DIR' does not exist"
    exit 1
fi

# Process each subdirectory
find "$TOP_DIR" -type d -mindepth 1 -maxdepth 1 | while read -r subdir; do
    # Get the subdirectory name
    dir_name=$(basename "$subdir")
    
    # Create corresponding analysis directory
    analysis_dir="${TOP_DIR}/${dir_name}_analysis"
    mkdir -p "$analysis_dir"
    
    echo "Processing: $subdir -> $analysis_dir"
    
    # Modify the Python script temporarily with new paths
    sed -i.bak "s|INPUT_FOLDER = .*|INPUT_FOLDER = \"$subdir\"|" "$PYTHON_SCRIPT"
    sed -i.bak "s|OUTPUT_FOLDER = .*|OUTPUT_FOLDER = \"$analysis_dir\"|" "$PYTHON_SCRIPT"
    
    # Run the analysis
    python "$PYTHON_SCRIPT"
    
    # Restore original script
    mv "${PYTHON_SCRIPT}.bak" "$PYTHON_SCRIPT"
done

echo "All subdirectories processed successfully" 