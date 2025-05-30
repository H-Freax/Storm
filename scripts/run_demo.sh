#!/bin/bash

# Try to initialize and activate conda environment if available
if command -v conda &> /dev/null; then
    # Conda is installed
    __conda_setup="$(conda 'shell.bash' 'hook' 2> /dev/null)"
    if [ $? -eq 0 ]; then
        eval "$__conda_setup"
        conda activate nanostorm
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/miniconda3/etc/profile.d/conda.sh"
        conda activate nanostorm
    fi
    unset __conda_setup

    # Test if conda environment is active and working
    if python -c "import sys; print('Conda Python:', sys.executable)" &> /dev/null; then
        echo "Conda is working and environment 'nanostorm' is active."
    else
        echo "Conda found, but failed to activate 'nanostorm' or run Python."
        echo "Falling back to system Python."
    fi
else
    echo "Conda not found. Using system Python environment."
fi

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Create logs directory and redirect all output to timestamped log file
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/run_demo_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# Add src directory to Python path
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Load environment variables from .env file
if [ -f .env ]; then
    source .env
else
    echo "Error: .env file not found"
    exit 1
fi

# Check for OpenAI API Key
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${YELLOW}Warning: OPENAI_API_KEY environment variable not set.${NC}"
    echo -e "Please set it with: ${GREEN}export OPENAI_API_KEY='your-key-here'${NC}"
    exit 1
fi

# Display banner
echo -e "${BLUE}=======================================${NC}"
echo -e "${BLUE}=      NanoStorm Demo Dialogue       =${NC}"
echo -e "${BLUE}=======================================${NC}"
echo

# Models to use
MODELS=(
  "anthropic/claude-3.7-sonnet"
  "openai/gpt-4o-mini"
  "google/gemini-2.5-flash-preview"
  "meta-llama/llama-3.3-70b-instruct"
)

# Profile directories to process
PROFILE_DIRS=(
  "profiles/users/basic"
  "profiles/users/unknown_40percent"
  "profiles/users/unknown_60percent"
  "profiles/users/unknown_80percent"
)

# RAG storage configurations (storage_name:storage_dir:storage_type)
RAG_CONFIGS=(
  "default:dialogue_vectors:qdrant"
)

# Default values
DEFAULT_NUM_TURNS=15
DEFAULT_DIALOGUES_DIR="dialogues"
DEFAULT_RAG=false
DEFAULT_SHARE_PROFILE=false
DEFAULT_STORAGE_DIR="dialogue_vectors"
DEFAULT_STORAGE_TYPE="qdrant"
DEFAULT_THRESHOLD=0.7
DEFAULT_BATCH_SIZE=10

# Function to show usage
show_usage() {
    echo -e "${BLUE}Usage:${NC}"
    echo -e "  ./run_demo.sh [options]"
    echo
    echo -e "${BLUE}Options:${NC}"
    echo -e "  --num-turns N           Number of dialogue turns (default: $DEFAULT_NUM_TURNS)"
    echo -e "  --dialogues-dir DIR     Dialogues directory (default: $DEFAULT_DIALOGUES_DIR)"
    echo -e "  --rag                   Enable RAG mode"
    echo -e "  --share-profile         Share user profile with assistant"
    echo -e "  --storage-dir DIR       Directory for vector storage (default: $DEFAULT_STORAGE_DIR)"
    echo -e "  --storage-type TYPE     Vector storage type (qdrant or milvus) (default: $DEFAULT_STORAGE_TYPE)"
    echo -e "  --threshold N           Similarity threshold (default: $DEFAULT_THRESHOLD)"
    echo -e "  --batch-size N          Batch size for import (default: $DEFAULT_BATCH_SIZE)"
    echo -e "  --help                  Show this help message"
    echo
    echo -e "${BLUE}Available Models:${NC}"
    for model in "${MODELS[@]}"; do
        echo -e "  - $model"
    done
    echo
    echo -e "${BLUE}Available Profile Directories:${NC}"
    for dir in "${PROFILE_DIRS[@]}"; do
        echo -e "  - $dir"
    done
    echo
    echo -e "${BLUE}Available RAG Configurations:${NC}"
    for config in "${RAG_CONFIGS[@]}"; do
        echo -e "  - $config"
    done
}

# Function to clean storage directories
clean_storage() {
    local storage_dir="$1"
    
    echo -e "${YELLOW}Cleaning storage directory: $storage_dir${NC}"
    if [ -d "$storage_dir" ]; then
        rm -rf "$storage_dir"/*
        rm -rf "$storage_dir"/.lock
        rm -rf "$storage_dir"/.qdrant
    else
        mkdir -p "$storage_dir"
    fi
}

# Function to import dialogues to vector storage
import_dialogues() {
    local import_dir="$1"
    local storage_dir="$2"
    local storage_type="$3"
    local batch_size="$4"
    local threshold="$5"
    
    echo -e "${GREEN}Starting dialogue import from ${import_dir} to ${storage_type} storage (${storage_dir})...${NC}"
    
    # Create storage directory if it doesn't exist
    mkdir -p "$storage_dir"
    
    # Run the import script
    python src/rag/import_json_to_rag.py \
        --dialogues-dir "$import_dir" \
        --storage-dir "$storage_dir" \
        --vector-storage "$storage_type" \
        --batch-size "$batch_size" \
        --threshold "$threshold"
        
    return $?
}

# Function to test a query
test_query() {
    local import_dir="$1"
    local storage_dir="$2"
    local storage_type="$3"
    local threshold="$4"
    local query="$5"
    
    echo -e "${GREEN}Testing query: ${query}${NC}"
    python src/rag/import_json_to_rag.py \
        --dialogues-dir "$import_dir" \
        --storage-dir "$storage_dir" \
        --vector-storage "$storage_type" \
        --threshold "$threshold" \
        --query "$query"
        
    return $?
}

# Function to install dependencies
install_dependencies() {
    local storage_type="$1"
    
    echo -e "${GREEN}Checking dependencies...${NC}"
    pip install -q tqdm

    if [ "$storage_type" = "qdrant" ]; then
        pip install -q qdrant-client
    elif [ "$storage_type" = "milvus" ]; then
        pip install -q pymilvus
    fi
}

# Function to log configurations
log_configurations() {
    local log_file="run_demo_log.txt"
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo "Timestamp: $timestamp" >> "$log_file"
    echo "Selected Models: ${selected_models[@]}" >> "$log_file"
    echo "Selected Profile Directories: ${selected_profiles[@]}" >> "$log_file"
    echo "RAG Mode: $RAG" >> "$log_file"
    if [ "$RAG" = true ]; then
        echo "Storage Directory: $STORAGE_DIR" >> "$log_file"
        echo "Storage Type: $STORAGE_TYPE" >> "$log_file"
        echo "Selected RAG Sources: ${selected_sources[@]}" >> "$log_file"
    fi
    echo "----------------------------------------" >> "$log_file"
}

# Function to find RAG data sources
find_rag_sources() {
    local base_dir="$1"
    local sources=()
    
    # Check if base directory exists
    if [ ! -d "$base_dir" ]; then
        echo -e "${YELLOW}Warning: $base_dir directory not found${NC}"
        return 1
    fi
    
    # Find all JSON files in the directory
    while IFS= read -r -d '' file; do
        if [[ "$file" == *.json ]]; then
            sources+=("$file")
        fi
    done < <(find "$base_dir" -type f -name "*.json" -print0)
    
    echo "${sources[@]}"
}

# Function to parse RAG source path
parse_rag_source() {
    local path="$1"
    local components=($(echo "$path" | tr '/' ' '))
    
    # Extract model name and type
    local model_name=""
    local model_type=""
    local source_number=""
    
    for component in "${components[@]}"; do
        if [[ "$component" == *"storm_json"* ]]; then
            if [[ "$component" == *"with"* ]]; then
                model_type="with_rag"
            elif [[ "$component" == *"without"* ]]; then
                model_type="without_rag"
            fi
        elif [[ "$component" =~ ^[0-9]+$ ]]; then
            source_number="$component"
        elif [[ "$component" != "example_data" ]]; then
            model_name="$component"
        fi
    done
    
    echo "$model_name:$model_type:$source_number"
}

# Function to group RAG sources
group_rag_sources() {
    local sources=("$@")
    local -A grouped
    
    for source in "${sources[@]}"; do
        local info=($(parse_rag_source "$source" | tr ':' ' '))
        local model_name="${info[0]}"
        local model_type="${info[1]}"
        local source_number="${info[2]}"
        
        if [ -n "$model_name" ] && [ -n "$model_type" ]; then
            grouped["$model_name:$model_type"]+="$source_number:$source "
        fi
    done
    
    echo "${!grouped[@]}"
}

# Function to read lines into an array
read_lines_into_array() {
    local -a array=()
    while IFS= read -r line; do
        array+=("$line")
    done
    echo "${array[@]}"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --num-turns)
            NUM_TURNS="$2"
            shift
            shift
            ;;
        --dialogues-dir)
            DIALOGUES_DIR="$2"
            shift
            shift
            ;;
        --rag)
            RAG=true
            shift
            ;;
        --share-profile)
            SHARE_PROFILE=true
            shift
            ;;
        --storage-dir)
            STORAGE_DIR="$2"
            shift
            shift
            ;;
        --storage-type)
            STORAGE_TYPE="$2"
            shift
            shift
            ;;
        --threshold)
            THRESHOLD="$2"
            shift
            shift
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo -e "${YELLOW}Unknown option: $1${NC}"
            show_usage
            exit 1
            ;;
    esac
done

# Set default values if not provided
NUM_TURNS=${NUM_TURNS:-$DEFAULT_NUM_TURNS}
DIALOGUES_DIR=${DIALOGUES_DIR:-$DEFAULT_DIALOGUES_DIR}
RAG=${RAG:-$DEFAULT_RAG}
SHARE_PROFILE=${SHARE_PROFILE:-$DEFAULT_SHARE_PROFILE}
STORAGE_DIR=${STORAGE_DIR:-$DEFAULT_STORAGE_DIR}
STORAGE_TYPE=${STORAGE_TYPE:-$DEFAULT_STORAGE_TYPE}
THRESHOLD=${THRESHOLD:-$DEFAULT_THRESHOLD}
BATCH_SIZE=${BATCH_SIZE:-$DEFAULT_BATCH_SIZE}

# Show available models
echo -e "${BLUE}Available Models:${NC}"
for i in "${!MODELS[@]}"; do
    echo -e "  $((i+1)). ${MODELS[$i]}"
done

# Get model selection
echo -e "${YELLOW}You can select multiple models by entering comma-separated numbers.${NC}"
echo -e "${YELLOW}For example: 1,3,4 will select the first, third, and fourth models.${NC}"
echo -e "${YELLOW}Enter your selections:${NC}"
read -p "> " model_choices
echo "Model choices selected: $model_choices"
IFS=',' read -ra selected_model_indices <<< "$model_choices"
selected_models=()

for index in "${selected_model_indices[@]}"; do
    if ! [[ "$index" =~ ^[0-9]+$ ]] || [ "$index" -lt 1 ] || [ "$index" -gt "${#MODELS[@]}" ]; then
        echo -e "${RED}Invalid model selection: $index${NC}"
        continue
    fi
    selected_models+=("${MODELS[$((index-1))]}")
done

if [ ${#selected_models[@]} -eq 0 ]; then
    echo -e "${RED}No valid models selected${NC}"
    exit 1
fi

echo
echo -e "${GREEN}Selected Models:${NC}"
for model in "${selected_models[@]}"; do
    echo -e "  - $model"
done

# Show available profile directories
echo
echo -e "${BLUE}Available Profile Directories:${NC}"
for i in "${!PROFILE_DIRS[@]}"; do
    echo -e "  $((i+1)). ${PROFILE_DIRS[$i]}"
done

# Get profile directory selection
echo -e "${YELLOW}You can select multiple profile directories by entering comma-separated numbers.${NC}"
echo -e "${YELLOW}For example: 1,3,4 will select the first, third, and fourth directories.${NC}"
echo -e "${YELLOW}Enter your selections:${NC}"
read -p "> " profile_choices
echo "Profile choices selected: $profile_choices"
IFS=',' read -ra selected_profile_indices <<< "$profile_choices"
selected_profiles=()

for index in "${selected_profile_indices[@]}"; do
    if ! [[ "$index" =~ ^[0-9]+$ ]] || [ "$index" -lt 1 ] || [ "$index" -gt "${#PROFILE_DIRS[@]}" ]; then
        echo -e "${RED}Invalid profile directory selection: $index${NC}"
        continue
    fi
    selected_profiles+=("${PROFILE_DIRS[$((index-1))]}")
done

if [ ${#selected_profiles[@]} -eq 0 ]; then
    echo -e "${RED}No valid profile directories selected${NC}"
    exit 1
fi

echo
echo -e "${GREEN}Selected Profile Directories:${NC}"
for profile in "${selected_profiles[@]}"; do
    echo -e "  - $profile"
done

# Ask for RAG mode
echo
read -p "Enable RAG mode? (y/n) [n]: " rag_choice
echo "RAG mode choice selected: $rag_choice"
RAG=${rag_choice:-n}
RAG=$([ "$RAG" = "y" ] || [ "$RAG" = "Y" ] && echo "true" || echo "false")

if [ "$RAG" = true ]; then
    echo
    echo -e "${BLUE}Available RAG Data Sources:${NC}"
    
    # Ask for custom RAG dataset path
    echo -e "${YELLOW}Enter the absolute path for the RAG dataset (default: example_data/storm_json_final):${NC}"
    read -p "> " custom_rag_path
    echo "Custom RAG dataset path selected: $custom_rag_path"
    RAG_BASE_DIR=${custom_rag_path:-"example_data/storm_json_final"}
    
    # Find and group RAG sources
    rag_sources=($(find_rag_sources "$RAG_BASE_DIR"))
    if [ ${#rag_sources[@]} -eq 0 ]; then
        echo -e "${RED}No JSON files found in $RAG_BASE_DIR directory${NC}"
        exit 1
    fi
    
    # Group sources by model and type
    grouped_sources=($(group_rag_sources "${rag_sources[@]}"))
    
    # Display grouped sources
    echo -e "${BLUE}Available Models and Types:${NC}"
    for i in "${!grouped_sources[@]}"; do
        local info=($(echo "${grouped_sources[$i]}" | tr ':' ' '))
        local model_name="${info[0]}"
        local model_type="${info[1]}"
        echo -e "  $((i+1)). $model_name ($model_type)"
    done

    # Use previously selected models and types
    echo -e "${YELLOW}Using previously selected models and types for RAG data sources.${NC}"
    selected_sources=("${grouped_sources[@]}")

    # Display selected sources
    echo -e "${GREEN}Selected RAG sources:${NC}"
    for source in "${selected_sources[@]}"; do
        echo -e "  - $source"
    done
    
    # Show available RAG configurations
    echo
    echo -e "${BLUE}Available RAG Configurations:${NC}"
    for i in "${!RAG_CONFIGS[@]}"; do
        echo -e "  $((i+1)). ${RAG_CONFIGS[$i]}"
    done

    # Get RAG configuration selection
    read -p "Select a RAG configuration (1-${#RAG_CONFIGS[@]}): " rag_config_choice
    echo "RAG configuration choice selected: $rag_config_choice"
    if ! [[ "$rag_config_choice" =~ ^[0-9]+$ ]] || [ "$rag_config_choice" -lt 1 ] || [ "$rag_config_choice" -gt "${#RAG_CONFIGS[@]}" ]; then
        echo -e "${RED}Invalid RAG configuration selection${NC}"
        exit 1
    fi

    # Parse selected RAG configuration
    IFS=':' read -r rag_name storage_dir storage_type <<< "${RAG_CONFIGS[$((rag_config_choice-1))]}"
    STORAGE_DIR="$storage_dir"
    STORAGE_TYPE="$storage_type"

    # Install dependencies for the storage type
    install_dependencies "$STORAGE_TYPE"

    # Import dialogues from all selected sources
    echo
    echo -e "${BLUE}=======================================${NC}"
    echo -e "${BLUE}= Starting RAG Import Process =${NC}"
    echo -e "${BLUE}=======================================${NC}"
    
    # Ask about cleaning storage
    echo
    read -p "Clean storage directory before importing? (y/n) [n]: " clean_choice
    echo "Clean storage choice selected: $clean_choice"
    if [[ "$clean_choice" = "y" || "$clean_choice" = "Y" ]]; then
        clean_storage "$STORAGE_DIR"
    fi
    
    # Import dialogues to vector storage
    if ! import_dialogues "$RAG_BASE_DIR" "$STORAGE_DIR" "$STORAGE_TYPE" "$BATCH_SIZE" "$THRESHOLD"; then
        echo -e "${RED}Failed to import dialogues${NC}"
        exit 1
    fi

    echo -e "${BLUE}=======================================${NC}"
    echo -e "${BLUE}= RAG Import Process Completed =${NC}"
    echo -e "${BLUE}=======================================${NC}"
    

    

fi

# Ask for profile sharing
echo
read -p "Share user profile with assistant? (y/n) [n]: " share_choice
echo "Share profile choice selected: $share_choice"
SHARE_PROFILE=${share_choice:-n}
SHARE_PROFILE=$([ "$SHARE_PROFILE" = "y" ] || [ "$SHARE_PROFILE" = "Y" ] && echo "true" || echo "false")

# Store all configurations
configurations=()

# Build all configurations
for model in "${selected_models[@]}"; do
    for profile in "${selected_profiles[@]}"; do
        # Build base command
        CMD="python \"$PROJECT_ROOT/src/demo_dialogue_generation.py\" \
            --assistant-model \"$model\" \
            --num-turns \"$NUM_TURNS\" \
            --profiles-dir \"$profile\" \
            --dialogues-dir \"$DIALOGUES_DIR\""

        # Add optional parameters
        if [ "$RAG" = true ]; then
            CMD="$CMD --rag --storage-dir \"$STORAGE_DIR\" --vector-storage \"$STORAGE_TYPE\" --threshold \"$THRESHOLD\""
            # Add all selected sources
            for source in "${selected_sources[@]}"; do
                CMD="$CMD --import-dir \"$source\""
            done
        fi

        if [ "$SHARE_PROFILE" = true ]; then
            CMD="$CMD --share-profile"
        fi

        # Store configuration
        configurations+=("$model:$profile:$CMD")
    done
done

# Display all configurations
echo
echo -e "${BLUE}All Configurations:${NC}"
for i in "${!configurations[@]}"; do
    IFS=':' read -r model profile cmd <<< "${configurations[$i]}"
    echo -e "${BLUE}Configuration $((i+1)):${NC}"
    echo -e "  Model: ${GREEN}$model${NC}"
    echo -e "  Profile Directory: ${GREEN}$profile${NC}"
    echo -e "  Number of turns: ${GREEN}$NUM_TURNS${NC}"
    echo -e "  Dialogues directory: ${GREEN}$DIALOGUES_DIR${NC}"
    echo -e "  RAG mode: ${GREEN}$RAG${NC}"
    if [ "$RAG" = true ]; then
        echo -e "  Storage directory: ${GREEN}$STORAGE_DIR${NC}"
        echo -e "  Storage type: ${GREEN}$STORAGE_TYPE${NC}"
        echo -e "  Similarity threshold: ${GREEN}$THRESHOLD${NC}"
    fi
    echo -e "  Share profile: ${GREEN}$SHARE_PROFILE${NC}"
    echo
done

# Ask for confirmation
echo -e "${YELLOW}Options:${NC}"
echo -e "  1: Run all configurations"
echo -e "  2: Select specific configurations to run"
echo -e "  3: Cancel"
read -p "Enter your choice (1-3): " choice
echo "RAG run option choice selected: $choice"
case $choice in
    1)
        echo -e "${GREEN}Running all configurations...${NC}"
        for config in "${configurations[@]}"; do
            IFS=':' read -r model profile cmd <<< "$config"
            echo
            echo -e "${BLUE}Running configuration for:${NC}"
            echo -e "  Model: ${GREEN}$model${NC}"
            echo -e "  Profile Directory: ${GREEN}$profile${NC}"
            echo -e "${BLUE}Starting demo...${NC}"
            eval "$cmd"
        done
        ;;
    2)
        echo -e "${YELLOW}Enter the numbers of configurations to run (comma-separated, e.g., 1,3,4):${NC}"
        read -p "> " selected_configs
        echo "RAG selected configurations: $selected_configs"
        IFS=',' read -ra config_indices <<< "$selected_configs"
        for index in "${config_indices[@]}"; do
            if ! [[ "$index" =~ ^[0-9]+$ ]] || [ "$index" -lt 1 ] || [ "$index" -gt "${#configurations[@]}" ]; then
                echo -e "${RED}Invalid configuration number: $index${NC}"
                continue
            fi
            
            config="${configurations[$((index-1))]}"
            IFS=':' read -r model profile cmd <<< "$config"
            echo
            echo -e "${BLUE}Running configuration for:${NC}"
            echo -e "  Model: ${GREEN}$model${NC}"
            echo -e "  Profile Directory: ${GREEN}$profile${NC}"
            echo -e "${BLUE}Starting demo...${NC}"
            eval "$cmd"
        done
        ;;
    3)
        echo -e "${YELLOW}Operation cancelled${NC}"
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo
echo -e "${GREEN}All selected configurations completed!${NC}"