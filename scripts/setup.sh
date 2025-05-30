#!/bin/bash

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Create conda environment if it doesn't exist
if ! conda env list | grep -q "nanostorm"; then
    echo "Creating conda environment..."
    conda create -n nanostorm python=3.9 -y
fi

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate nanostorm

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r "$PROJECT_ROOT/requirements.txt"


# Set up environment variables
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    echo "Creating .env file..."
    echo "OPENAI_API_KEY=your_api_key_here" > "$PROJECT_ROOT/.env"
    echo "Please update the OPENAI_API_KEY in .env file with your actual API key"
fi

echo "Setup completed successfully!" 