#!/bin/bash

# Script to run summary analysis for all dialogue folders in the output directory
# Uses summary_analysis.py to generate per-file summaries

# Base directories
OUTPUT_DIR="./example_data/storm_json_final"
PYTHON_SCRIPT="./src/analysis/summary_analysis.py"

# Find all model dialogues directories
find "$OUTPUT_DIR" -type d -name "*_dialogues" | while read -r model_dialogues_dir; do
    # Extract model name (remove suffix)
    model_name=$(basename "$model_dialogues_dir" | sed 's/_dialogues//')
    
    # Process each dataset under the model dialogues
    find "$model_dialogues_dir" -type d -mindepth 1 | while read -r dataset_dir; do
        dataset_name=$(basename "$dataset_dir")
        
        # Prepare output directory for summaries
        output_dir="$OUTPUT_DIR/${model_name}_turn_summaries/$dataset_name/$model_name"
        mkdir -p "$output_dir"
        
        echo "Generating summary: $dataset_dir -> $output_dir"
        
        # Update script paths
        sed -i.bak "s|INPUT_FOLDER = .*|INPUT_FOLDER = \"$dataset_dir\"|" "$PYTHON_SCRIPT"
        sed -i.bak "s|OUTPUT_FOLDER = .*|OUTPUT_FOLDER = \"$output_dir\"|" "$PYTHON_SCRIPT"
        
        # Run summary analysis
        python "$PYTHON_SCRIPT"
        
        # Restore original script
        mv "${PYTHON_SCRIPT}.bak" "$PYTHON_SCRIPT"
    done
done

echo "All summary analyses completed successfully" 