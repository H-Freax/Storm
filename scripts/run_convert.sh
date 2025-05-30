#!/bin/bash

# Check if Python script exists
if [ ! -f "src/analysis/convert_to_profile.py" ]; then
    echo "Error: convert_to_profile.py not found in src/analysis/"
    exit 1
fi

# Function to show usage
show_usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -i, --input <path>       Input directory (default: example_data/storm_json_final)"
    echo "  -o, --output <path>      Output directory (default: example_data/storm_json_final_addprofiles)"
    echo "  -p, --profile <path>    Default profile directory (default: example_data/profiles/users/basic)"
    echo "  -h, --help              Show this help message"
}

# Default input directory
INPUT="dialogues"
OUTPUT="dialogues/storm_rag_addprofiles"
PROFILE="example_data/profiles/users/basic"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT="$2"
            shift 2
            ;;
        -p|--profile)
            PROFILE="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Check if input directory exists
if [ ! -d "$INPUT" ]; then
    echo "Error: Input directory '$INPUT' does not exist"
    exit 1
fi

# Build the command
CMD="python src/analysis/convert_to_profile.py $INPUT --all"

if [ ! -z "$OUTPUT" ]; then
    CMD="$CMD --output $OUTPUT"
fi

if [ ! -z "$PROFILE" ]; then
    CMD="$CMD --default-profile $PROFILE"
fi

# Show what will be done
echo "=== Profile Addition Process ==="
echo "Input directory: $INPUT"
echo "Output directory: $OUTPUT"
echo "Profile directory: $PROFILE"
echo
echo "This script will:"
echo "1. Scan all JSON files in '$INPUT' and its subdirectories"
echo "2. Count files that already have user profiles"
echo "3. Add user profiles to files that don't have them"
echo "4. Save results to '$OUTPUT'"
echo "5. Show detailed statistics about the process"
echo

# Ask for confirmation
read -p "Do you want to continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Operation cancelled"
    exit 1
fi

# Execute the command
echo "Starting process..."
echo "Running: $CMD"
eval $CMD

# Check if the process was successful
if [ $? -eq 0 ]; then
    echo
    echo "=== Process Completed Successfully ==="
    echo "Check the output above for detailed statistics"
else
    echo
    echo "=== Process Completed with Errors ==="
    echo "Some files may have failed to process"
    echo "Check conversion_failures.log for details"
fi 