#!/bin/bash

# Try to initialize and activate conda environment if available
if command -v conda &> /dev/null; then
    # Conda is installed
    __conda_setup="$(conda 'shell.bash' 'hook' 2> /dev/null)"
    if [ $? -eq 0 ]; then
        eval "$__conda_setup"
        conda activate graphrag
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/miniconda3/etc/profile.d/conda.sh"
        conda activate graphrag
    fi
    unset __conda_setup

    # Test if conda environment is active and working
    if python -c "import sys; print('Conda Python:', sys.executable)" &> /dev/null; then
        echo "Conda is working and environment 'graphrag' is active."
    else
        echo "Conda found, but failed to activate 'graphrag' or run Python."
        echo "Falling back to system Python."
    fi
else
    echo "Conda not found. Using system Python environment."
fi

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Display banner
echo -e "${BLUE}=======================================${NC}"
echo -e "${BLUE}= NanoStorm Dialogue RAG Import Tool =${NC}"
echo -e "${BLUE}=======================================${NC}"
echo

# Check for OpenAI API Key
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${YELLOW}Warning: OPENAI_API_KEY environment variable not set.${NC}"
    echo -e "Please set it with: ${GREEN}export OPENAI_API_KEY='your-key-here'${NC}"
    exit 1
fi

# Default values
# DEFAULT_IMPORT_DIR="./output_without_profile/claude/unknown_80percent/anthropic_claude-3.7-sonnet"
DEFAULT_IMPORT_DIR="./example_data/test"
DEFAULT_STORAGE_DIR="dialogue_vectors"
DEFAULT_STORAGE_TYPE="qdrant"
DEFAULT_BATCH_SIZE=10
DEFAULT_THRESHOLD=0.7
DEFAULT_NUM_TURNS=15
DEFAULT_MAX_CONCURRENT=4
DEFAULT_OUTPUT_DIR="./output/rag_results"

# OpenAI API keys rotation (for better concurrency with OpenAI models)
OPENAI_API_KEYS=(
  # Add more keys as needed
)

# Default models to use
MODELS=(
  "anthropic/claude-3.7-sonnet"
  "openai/gpt-4o-mini"
  "google/gemini-2.5-flash-preview"
  "meta-llama/llama-3.3-70b-instruct"
)

# Default profile directories
PROFILE_DIRS=(
  "profiles/users/basic"
  "profiles/users/unknown_40percent"
  "profiles/users/unknown_60percent"
  # "profiles/users/unknown_80percent"
)

# Default RAG storage configurations (storage_name:storage_dir:storage_type)
RAG_CONFIGS=(
  "default:dialogue_vectors:qdrant"
  # "milvus_store:milvus_vectors:milvus"
)

# Flags
RUN_BATCH=false
IMPORT_ONLY=false
QUERY_ONLY=false
GENERATE_ONLY=false
SKIP_IMPORT=false
SKIP_PROMPT=false
CLEAN_BEFORE_RUN=false
CLEAN_AFTER_RUN=false
REUSE_IMPORTED=false
MAX_FAILURES=3

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --import-dir)
            DEFAULT_IMPORT_DIR="$2"
            shift
            shift
            ;;
        --storage-dir)
            DEFAULT_STORAGE_DIR="$2"
            shift
            shift
            ;;
        --storage-type)
            DEFAULT_STORAGE_TYPE="$2"
            shift
            shift
            ;;
        --output-dir)
            DEFAULT_OUTPUT_DIR="$2"
            shift
            shift
            ;;
        --batch-size)
            DEFAULT_BATCH_SIZE="$2"
            shift
            shift
            ;;
        --threshold)
            DEFAULT_THRESHOLD="$2"
            shift
            shift
            ;;
        --num-turns)
            DEFAULT_NUM_TURNS="$2"
            shift
            shift
            ;;
        --max-concurrent)
            DEFAULT_MAX_CONCURRENT="$2"
            shift
            shift
            ;;
        --models)
            IFS=',' read -r -a MODELS <<< "$2"
            shift
            shift
            ;;
        --profiles)
            IFS=',' read -r -a PROFILE_DIRS <<< "$2"
            shift
            shift
            ;;
        --rag-configs)
            IFS=',' read -r -a RAG_CONFIGS <<< "$2"
            shift
            shift
            ;;
        --run-batch)
            RUN_BATCH=true
            shift
            ;;
        --import-only)
            IMPORT_ONLY=true
            shift
            ;;
        --query-only)
            QUERY_ONLY=true
            shift
            ;;
        --generate-only)
            GENERATE_ONLY=true
            shift
            ;;
        --skip-import)
            SKIP_IMPORT=true
            shift
            ;;
        --skip-prompt)
            SKIP_PROMPT=true
            shift
            ;;
        --clean-before)
            CLEAN_BEFORE_RUN=true
            shift
            ;;
        --clean-after)
            CLEAN_AFTER_RUN=true
            shift
            ;;
        --reuse-imported)
            REUSE_IMPORTED=true
            shift
            ;;
        --max-failures)
            MAX_FAILURES="$2"
            shift
            shift
            ;;
        --query)
            TEST_QUERY="$2"
            shift
            shift
            ;;
        *)
            echo -e "${YELLOW}Unknown argument: $1${NC}"
            shift
            ;;
    esac
done

# Function to show usage
show_usage() {
    echo -e "${BLUE}Usage:${NC}"
    echo -e "  ./import_and_test_rag.sh [options]"
    echo
    echo -e "${BLUE}Options:${NC}"
    echo -e "  --import-dir DIR        Source directory for dialogue imports (default: $DEFAULT_IMPORT_DIR)"
    echo -e "  --storage-dir DIR       Directory for vector storage (default: $DEFAULT_STORAGE_DIR)"
    echo -e "  --storage-type TYPE     Vector storage type (qdrant or milvus) (default: $DEFAULT_STORAGE_TYPE)"
    echo -e "  --output-dir DIR        Output directory for generated dialogues (default: $DEFAULT_OUTPUT_DIR)"
    echo -e "  --batch-size N          Batch size for import (default: $DEFAULT_BATCH_SIZE)"
    echo -e "  --threshold N           Similarity threshold (default: $DEFAULT_THRESHOLD)"
    echo -e "  --num-turns N           Number of dialogue turns (default: $DEFAULT_NUM_TURNS)"
    echo -e "  --max-concurrent N      Maximum concurrent processes (default: $DEFAULT_MAX_CONCURRENT)"
    echo -e "  --models \"M1,M2,...\"    Comma-separated list of models to use"
    echo -e "  --profiles \"P1,P2,...\"  Comma-separated list of profile directories"
    echo -e "  --rag-configs \"R1,R2\"   Comma-separated list of RAG configs (format: name:dir:type)"
    echo -e "  --run-batch             Run batch processing"
    echo -e "  --import-only           Only import dialogues, don't generate or query"
    echo -e "  --query-only            Only run query testing, don't generate dialogues"
    echo -e "  --generate-only         Only generate dialogues, don't run query testing"
    echo -e "  --skip-import           Skip import phase"
    echo -e "  --skip-prompt           Skip interactive prompts"
    echo -e "  --clean-before          Clean storage directories before importing"
    echo -e "  --clean-after           Clean storage directories after batch is complete"
    echo -e "  --reuse-imported        Reuse the same imported vectors for all models/profiles"
    echo -e "  --max-failures N        Maximum failures before skipping a model (default: $MAX_FAILURES)"
    echo -e "  --query \"TEXT\"          Test query text"
    echo
    echo -e "${BLUE}Examples:${NC}"
    echo -e "  # Run interactive mode with defaults"
    echo -e "  ./import_and_test_rag.sh"
    echo
    echo -e "  # Import only from specific directory"
    echo -e "  ./import_and_test_rag.sh --import-only --import-dir ./my_dialogues/"
    echo
    echo -e "  # Run batch processing with multiple models and profiles"
    echo -e "  ./import_and_test_rag.sh --run-batch --models \"anthropic/claude-3.7-sonnet,openai/gpt-4o-mini\" \\"
    echo -e "    --profiles \"profiles/users/basic,profiles/users/unknown_40percent\" \\"
    echo -e "    --reuse-imported --clean-before"
    echo
    echo -e "  # Test a specific query on existing database"
    echo -e "  ./import_and_test_rag.sh --skip-import --query \"How does RAG improve dialogue generation?\""
    echo
}

# Function to clean storage directories
clean_storage() {
    local storage_dir="$1"
    
    echo -e "${YELLOW}Cleaning storage directory: $storage_dir${NC}"
    if [ -d "$storage_dir" ]; then
        rm -rf "$storage_dir"/*
        rm -rf "$storage_dir"/.lock
    else
        mkdir -p "$storage_dir"
    fi
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

# Function to generate dialogue with RAG
generate_dialogue() {
    local import_dir="$1"
    local storage_dir="$2"
    local storage_type="$3"
    local threshold="$4"
    local num_turns="$5"
    local model="$6"
    local profile_dir="$7"
    local output_dir="$8"
    
    # Create output directory if specified
    if [ ! -z "$output_dir" ]; then
        mkdir -p "$output_dir"
        output_param="--output-dir $output_dir"
    else
        output_param=""
    fi
    
    # Set model param if specified
    local model_param=""
    if [ ! -z "$model" ]; then
        model_param="--assistant-model $model"
    fi
    
    # Set profile param if specified
    local profile_param=""
    if [ ! -z "$profile_dir" ]; then
        profile_param="--profiles-dir $profile_dir"
    fi
    
    echo -e "${GREEN}Generating dialogue with ${num_turns} turns using RAG...${NC}"
    echo -e "${GREEN}Model: ${model:-default}, Profile: ${profile_dir:-default}${NC}"
    echo -e "${GREEN}Output directory: ${output_dir:-default}${NC}"
    
    python src/demo_dialogue_generation.py \
        --rag \
        --dialogues-dir "$import_dir" \
        --storage-dir "$storage_dir" \
        --vector-storage "$storage_type" \
        --threshold "$threshold" \
        --num-turns "$num_turns" \
        $model_param $profile_param $output_param
        
    return $?
}

# Function to parse RAG config
parse_rag_config() {
    local config="$1"
    local IFS=":"
    read -r name dir type <<< "$config"
    echo "$name $dir $type"
}

# Function to generate output directory path for a specific config
get_output_dir() {
    local base_dir="$1"
    local rag_name="$2"
    local model="$3"
    local profile_type="$4"
    
    # Sanitize model name for directory
    local model_dir=$(echo "$model" | sed 's|/|_|g' | sed 's|:|_|g')
    
    # Extract profile type from directory path
    local profile_name=$(basename "$profile_type")
    
    echo "${base_dir}/${rag_name}/${model_dir}/${profile_name}"
}

# Batch processing function
run_batch_process() {
    local total_configs=$((${#MODELS[@]} * ${#PROFILE_DIRS[@]} * (REUSE_IMPORTED ? 1 : ${#RAG_CONFIGS[@]})))
    local current_config=0
    local total_failures=0
    
    echo -e "${BLUE}=======================================${NC}"
    echo -e "${BLUE}=  Starting Batch Processing (${total_configs} configs) =${NC}"
    echo -e "${BLUE}=======================================${NC}"
    
    # If we're reusing the imported data, we'll process differently
    if [ "$REUSE_IMPORTED" = true ]; then
        # We'll only use the first RAG config for import
        local rag_config="${RAG_CONFIGS[0]}"
        read -r rag_name storage_dir storage_type <<< "$(parse_rag_config "$rag_config")"
        
        echo -e "${BLUE}Using RAG config: $rag_name (reusing for all models/profiles)${NC}"
        
        # Clean storage if requested
        if [ "$CLEAN_BEFORE_RUN" = true ]; then
            clean_storage "$storage_dir"
        fi
        
        # Install dependencies for the storage type
        install_dependencies "$storage_type"
        
        # Import dialogues to vector storage if not skipped
        if [ "$SKIP_IMPORT" = false ]; then
            if ! import_dialogues "$DEFAULT_IMPORT_DIR" "$storage_dir" "$storage_type" "$DEFAULT_BATCH_SIZE" "$DEFAULT_THRESHOLD"; then
                echo -e "${RED}Failed to import dialogues for RAG config: $rag_name${NC}"
                return 1
            fi
        else
            echo -e "${YELLOW}Skipping import phase as requested...${NC}"
        fi
        
        # If only import was requested, we're done
        if [ "$IMPORT_ONLY" = true ]; then
            echo -e "${GREEN}Import completed for RAG config: $rag_name${NC}"
            return 0
        fi
        
        # Run test query if provided
        if [ ! -z "$TEST_QUERY" ]; then
            echo -e "${GREEN}Running test query for RAG config: $rag_name${NC}"
            test_query "$DEFAULT_IMPORT_DIR" "$storage_dir" "$storage_type" "$DEFAULT_THRESHOLD" "$TEST_QUERY"
        fi
        
        # If only query was requested, we're done
        if [ "$QUERY_ONLY" = true ]; then
            echo -e "${GREEN}Query testing completed for RAG config: $rag_name${NC}"
            return 0
        fi
        
        # Iterate through each model and profile to generate dialogues
        for model in "${MODELS[@]}"; do
            # Model failure counter
            local model_failures=0
            
            # Iterate through each profile directory
            for profile_dir in "${PROFILE_DIRS[@]}"; do
                # Skip if we've seen too many failures with this model
                if [ "$model_failures" -ge "$MAX_FAILURES" ]; then
                    echo -e "${RED}⚠️ Skipping model $model due to repeated failures${NC}"
                    break
                fi
                
                # Increment config counter
                current_config=$((current_config + 1))
                
                # Extract profile type from directory path
                local profile_type=$(basename "$profile_dir")
                
                # Create output directory
                local output_dir=$(get_output_dir "$DEFAULT_OUTPUT_DIR" "$rag_name" "$model" "$profile_type")
                
                echo -e "${BLUE}=======================================${NC}"
                echo -e "${BLUE}= Config $current_config/$total_configs =${NC}"
                echo -e "${BLUE}= RAG: $rag_name | Model: $model | Profile: $profile_type =${NC}"
                echo -e "${BLUE}=======================================${NC}"
                
                # Check if user wants to skip this config
                if [ "$SKIP_PROMPT" = false ]; then
                    echo -e "${YELLOW}Run this configuration? (y/n/a)${NC}"
                    echo -e "${YELLOW}  y: Yes, run this config${NC}"
                    echo -e "${YELLOW}  n: Skip this config${NC}"
                    echo -e "${YELLOW}  a: Run all remaining configs without prompting${NC}"
                    read answer
                    
                    if [ "$answer" = "n" ]; then
                        echo -e "${YELLOW}Skipping this configuration...${NC}"
                        continue
                    elif [ "$answer" = "a" ]; then
                        echo -e "${YELLOW}Running all remaining configurations without prompting...${NC}"
                        SKIP_PROMPT=true
                    fi
                fi
                
                # Generate dialogue with RAG
                echo -e "${GREEN}Generating dialogue with RAG for this config...${NC}"
                if ! generate_dialogue "$DEFAULT_IMPORT_DIR" "$storage_dir" "$storage_type" "$DEFAULT_THRESHOLD" "$DEFAULT_NUM_TURNS" "$model" "$profile_dir" "$output_dir"; then
                    echo -e "${RED}❌ Dialogue generation failed for this config${NC}"
                    model_failures=$((model_failures + 1))
                    total_failures=$((total_failures + 1))
                else
                    echo -e "${GREEN}✅ Dialogue generation completed successfully${NC}"
                fi
                
                echo -e "${GREEN}Configuration $current_config/$total_configs completed${NC}"
                echo
            done
        done
        
        # Clean up after run if requested
        if [ "$CLEAN_AFTER_RUN" = true ]; then
            clean_storage "$storage_dir"
        fi
    else
        # Process each RAG config separately
        for rag_config in "${RAG_CONFIGS[@]}"; do
            # Parse RAG config
            read -r rag_name storage_dir storage_type <<< "$(parse_rag_config "$rag_config")"
            
            # Clean storage if requested
            if [ "$CLEAN_BEFORE_RUN" = true ]; then
                clean_storage "$storage_dir"
            fi
            
            # Skip import if requested
            if [ "$SKIP_IMPORT" = false ]; then
                # Install dependencies for the storage type
                install_dependencies "$storage_type"
                
                # Import dialogues to vector storage
                if ! import_dialogues "$DEFAULT_IMPORT_DIR" "$storage_dir" "$storage_type" "$DEFAULT_BATCH_SIZE" "$DEFAULT_THRESHOLD"; then
                    echo -e "${RED}Failed to import dialogues for RAG config: $rag_name${NC}"
                    continue
                fi
            else
                echo -e "${YELLOW}Skipping import phase as requested...${NC}"
            fi
            
            # If only import was requested, continue to next RAG config
            if [ "$IMPORT_ONLY" = true ]; then
                echo -e "${GREEN}Import completed for RAG config: $rag_name${NC}"
                continue
            fi
            
            # Run test query if provided
            if [ ! -z "$TEST_QUERY" ]; then
                echo -e "${GREEN}Running test query for RAG config: $rag_name${NC}"
                test_query "$DEFAULT_IMPORT_DIR" "$storage_dir" "$storage_type" "$DEFAULT_THRESHOLD" "$TEST_QUERY"
            fi
            
            # If only query was requested, continue to next RAG config
            if [ "$QUERY_ONLY" = true ]; then
                echo -e "${GREEN}Query testing completed for RAG config: $rag_name${NC}"
                continue
            fi
            
            # Iterate through each model
            for model in "${MODELS[@]}"; do
                # Model failure counter
                local model_failures=0
                
                # Sanitize model name for directory
                local model_dir=$(echo "$model" | sed 's|/|_|g' | sed 's|:|_|g')
                
                # Iterate through each profile directory
                for profile_dir in "${PROFILE_DIRS[@]}"; do
                    # Skip if we've seen too many failures with this model
                    if [ "$model_failures" -ge "$MAX_FAILURES" ]; then
                        echo -e "${RED}⚠️ Skipping model $model due to repeated failures${NC}"
                        break
                    fi
                    
                    # Increment config counter
                    current_config=$((current_config + 1))
                    
                    # Extract profile type from directory path
                    local profile_type=$(basename "$profile_dir")
                    
                    # Create output directory
                    local output_dir=$(get_output_dir "$DEFAULT_OUTPUT_DIR" "$rag_name" "$model" "$profile_type")
                    
                    echo -e "${BLUE}=======================================${NC}"
                    echo -e "${BLUE}= Config $current_config/$total_configs =${NC}"
                    echo -e "${BLUE}= RAG: $rag_name | Model: $model | Profile: $profile_type =${NC}"
                    echo -e "${BLUE}=======================================${NC}"
                    
                    # Check if user wants to skip this config
                    if [ "$SKIP_PROMPT" = false ]; then
                        echo -e "${YELLOW}Run this configuration? (y/n/a)${NC}"
                        echo -e "${YELLOW}  y: Yes, run this config${NC}"
                        echo -e "${YELLOW}  n: Skip this config${NC}"
                        echo -e "${YELLOW}  a: Run all remaining configs without prompting${NC}"
                        read answer
                        
                        if [ "$answer" = "n" ]; then
                            echo -e "${YELLOW}Skipping this configuration...${NC}"
                            continue
                        elif [ "$answer" = "a" ]; then
                            echo -e "${YELLOW}Running all remaining configurations without prompting...${NC}"
                            SKIP_PROMPT=true
                        fi
                    fi
                    
                    # Generate dialogue with RAG
                    echo -e "${GREEN}Generating dialogue with RAG for this config...${NC}"
                    if ! generate_dialogue "$DEFAULT_IMPORT_DIR" "$storage_dir" "$storage_type" "$DEFAULT_THRESHOLD" "$DEFAULT_NUM_TURNS" "$model" "$profile_dir" "$output_dir"; then
                        echo -e "${RED}❌ Dialogue generation failed for this config${NC}"
                        model_failures=$((model_failures + 1))
                        total_failures=$((total_failures + 1))
                    else
                        echo -e "${GREEN}✅ Dialogue generation completed successfully${NC}"
                    fi
                    
                    echo -e "${GREEN}Configuration $current_config/$total_configs completed${NC}"
                    echo
                done
            done
            
            # Clean up after run if requested
            if [ "$CLEAN_AFTER_RUN" = true ]; then
                clean_storage "$storage_dir"
            fi
        done
    fi
    
    echo -e "${BLUE}=======================================${NC}"
    echo -e "${BLUE}= Batch Processing Summary =${NC}"
    echo -e "${BLUE}=======================================${NC}"
    echo -e "${GREEN}Total configurations processed: $current_config/${total_configs}${NC}"
    echo -e "${YELLOW}Total failures: $total_failures${NC}"
}

# Show usage if no arguments are provided
if [ $# -eq 0 ]; then
    show_usage
    echo -e "${YELLOW}Continuing with default values...${NC}"
    echo
fi

# Main execution logic
if [ "$RUN_BATCH" = true ]; then
    # Run batch processing
    run_batch_process
else
    # Single run mode
    # Install dependencies
    install_dependencies "$DEFAULT_STORAGE_TYPE"
    
    # Clean storage if requested
    if [ "$CLEAN_BEFORE_RUN" = true ]; then
        clean_storage "$DEFAULT_STORAGE_DIR"
    fi
    
    # Import dialogues if not skipped
    if [ "$SKIP_IMPORT" = false ]; then
        if ! import_dialogues "$DEFAULT_IMPORT_DIR" "$DEFAULT_STORAGE_DIR" "$DEFAULT_STORAGE_TYPE" "$DEFAULT_BATCH_SIZE" "$DEFAULT_THRESHOLD"; then
            echo -e "${RED}Failed to import dialogues${NC}"
            exit 1
        fi
    fi

# If import successful, offer to test queries
echo
echo -e "${GREEN}Import completed.${NC}"
echo
    
    # If a test query was provided as argument, run it
    if [ ! -z "$TEST_QUERY" ]; then
        test_query "$DEFAULT_IMPORT_DIR" "$DEFAULT_STORAGE_DIR" "$DEFAULT_STORAGE_TYPE" "$DEFAULT_THRESHOLD" "$TEST_QUERY"
    elif [ "$QUERY_ONLY" = false ] && [ "$GENERATE_ONLY" = false ]; then
        # If no specific mode is requested, offer interactive query testing
echo -e "${YELLOW}Would you like to test a query? (y/n)${NC}"
read answer

if [[ "$answer" == "y" || "$answer" == "Y" ]]; then
    echo -e "${YELLOW}Enter your test query:${NC}"
    read query
    
            test_query "$DEFAULT_IMPORT_DIR" "$DEFAULT_STORAGE_DIR" "$DEFAULT_STORAGE_TYPE" "$DEFAULT_THRESHOLD" "$query"
        fi
    fi

    # If not in query-only mode, offer to generate dialogue
    if [ "$QUERY_ONLY" = false ]; then
        if [ "$GENERATE_ONLY" = true ]; then
            # Automatically generate dialogue in generate-only mode
            generate_dialogue "$DEFAULT_IMPORT_DIR" "$DEFAULT_STORAGE_DIR" "$DEFAULT_STORAGE_TYPE" "$DEFAULT_THRESHOLD" "$DEFAULT_NUM_TURNS" "" "" ""
        else
            # Interactive mode: ask user if they want to generate dialogue
echo
echo -e "${YELLOW}Would you like to generate a dialogue with RAG? (y/n)${NC}"
read answer

if [[ "$answer" == "y" || "$answer" == "Y" ]]; then
                echo -e "${YELLOW}Enter number of turns (default: $DEFAULT_NUM_TURNS):${NC}"
    read num_turns
    
    if [ -z "$num_turns" ]; then
                    num_turns=$DEFAULT_NUM_TURNS
                fi
                
                generate_dialogue "$DEFAULT_IMPORT_DIR" "$DEFAULT_STORAGE_DIR" "$DEFAULT_STORAGE_TYPE" "$DEFAULT_THRESHOLD" "$num_turns" "" "" ""
            fi
        fi
    fi
    
    # Clean up after run if requested
    if [ "$CLEAN_AFTER_RUN" = true ]; then
        clean_storage "$DEFAULT_STORAGE_DIR"
    fi
fi

echo
echo -e "${GREEN}All tasks completed!${NC}" 