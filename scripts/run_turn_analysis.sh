#!/bin/bash

# Script to analyze all dialogue folders in output directory
# Automatically processes each model's data following the pattern in turn_pairs_analysis.py

# Base directories
OUTPUT_DIR="./example_data/storm_json_final"
PYTHON_SCRIPT="./src/analysis/turn_pairs_analysis.py"

# Find all dialogue directories
find "$OUTPUT_DIR" -type d -name "*_dialogues" | while read -r model_dialogues_dir; do
    # Extract model name from directory
    model_name=$(basename "$model_dialogues_dir" | sed 's/_dialogues//')
    
    # Process each dataset subdirectory
    find "$model_dialogues_dir" -type d -mindepth 1 | while read -r dataset_dir; do
        dataset_name=$(basename "$dataset_dir")
        
        # Create output directory
        output_dir="$OUTPUT_DIR/${model_name}_turn_analyses/$dataset_name/$model_name"
        mkdir -p "$output_dir"
        
        echo "Processing: $dataset_dir -> $output_dir"
        
        # Modify the Python script temporarily with new paths
        sed -i.bak "s|INPUT_FOLDER = .*|INPUT_FOLDER = \"$dataset_dir\"|" "$PYTHON_SCRIPT"
        sed -i.bak "s|OUTPUT_FOLDER = .*|OUTPUT_FOLDER = \"$output_dir\"|" "$PYTHON_SCRIPT"
        
        # Run the analysis
        python "$PYTHON_SCRIPT"
        
        # Restore original script
        mv "${PYTHON_SCRIPT}.bak" "$PYTHON_SCRIPT"
    done
done

echo "All folders processed successfully"
