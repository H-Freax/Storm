#!/bin/bash

# Script to run summary analysis for all subdirectories in a given top-level directory
# Creates corresponding summary folders for each subdirectory

# Base directories
DEFAULT_TOP_DIR="./example_data/storm_json_final"
TOP_DIR="${1:-$DEFAULT_TOP_DIR}"
PYTHON_SCRIPT="./src/analysis/summary_analysis.py"

# Top-level summary directory
SUMMARY_TOP_DIR="${TOP_DIR}_summary"
mkdir -p "$SUMMARY_TOP_DIR"

# Check if the input directory exists
if [ ! -d "$TOP_DIR" ]; then
    echo "Error: Directory '$TOP_DIR' does not exist"
    exit 1
fi

# List subdirectories to be processed
echo "Folders to be processed (excluding *_analysis and *_summary):"
find "$TOP_DIR" -mindepth 1 -maxdepth 1 -type d ! -name "*_analysis" ! -name "*_summary" | sed 's|.*/||'

# Process each subdirectory
find "$TOP_DIR" -mindepth 1 -maxdepth 1 -type d ! -name "*_analysis" ! -name "*_summary" | while read -r subdir; do
    dir_name=$(basename "$subdir")
    summary_dir="${SUMMARY_TOP_DIR}/${dir_name}_summary"
    mkdir -p "$summary_dir"

    echo "Processing: $subdir -> $summary_dir"

    # Update script paths
    sed -i.bak "s|INPUT_FOLDER = .*|INPUT_FOLDER = \"$subdir\"|" "$PYTHON_SCRIPT"
    sed -i.bak "s|OUTPUT_FOLDER = .*|OUTPUT_FOLDER = \"$summary_dir\"|" "$PYTHON_SCRIPT"

    # Run the summary analysis
    python "$PYTHON_SCRIPT"

    # Restore original script
    mv "${PYTHON_SCRIPT}.bak" "$PYTHON_SCRIPT"
done

echo "All summary analyses completed successfully" 