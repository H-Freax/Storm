#!/bin/bash

# Run profile combination script
echo "Running profile combination script..."
python src/profile_generate.py

# Check if the first script executed successfully
if [ $? -eq 0 ]; then
    echo "Profile combination completed successfully"
    echo "Running dialogue generation script..."
    python src/profile_generate.py
    
    # Check if the second script executed successfully
    if [ $? -eq 0 ]; then
        echo "Dialogue generation completed successfully"
        echo "Running unknown profiles generation script..."
        python src/profile_generators/generate_unknown_profiles.py
    else
        echo "Dialogue generation failed, stopping execution"
        exit 1
    fi
else
    echo "Profile combination failed, stopping execution"
    exit 1
fi 