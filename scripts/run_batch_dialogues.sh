#!/bin/bash

# Load environment variables from .env file
if [ -f .env ]; then
    source .env
else
    echo "Error: .env file not found"
    exit 1
fi

# Profile directories to process
PROFILE_DIRS=(
  "profiles/users/basic"
  "profiles/users/unknown_40percent"
  "profiles/users/unknown_60percent"
  "profiles/users/unknown_80percent"
)

# Models to use
MODELS=(
  "anthropic/claude-3.7-sonnet"
  "openai/gpt-4o-mini"
  "google/gemini-2.5-flash-preview"
  "meta-llama/llama-3.3-70b-instruct"
)

# Number of concurrent processes
MAX_CONCURRENT=15
# Number of dialogue turns
NUM_TURNS=15

# Flag to skip prompts and run all batches
RUN_ALL=false

# Max failures before skipping a model
MAX_FAILURES=3



# Iterate through each model
for MODEL in "${MODELS[@]}"; do
  # Skip counter for this model
  MODEL_FAILURES=0
  
  # Sanitize model name for directory
  MODEL_DIR=$(echo "$MODEL" | sed 's|/|_|g' | sed 's|:|_|g')
  
  # Iterate through each profile directory
  for PROFILE_DIR in "${PROFILE_DIRS[@]}"; do
    # Check if we've seen too many failures with this model
    if [ "$MODEL_FAILURES" -ge "$MAX_FAILURES" ]; then
      echo "⚠️ Skipping model $MODEL due to repeated failures"
      break
    fi
    
    # Extract profile type from directory path
    PROFILE_TYPE=$(basename "$PROFILE_DIR")
    
    # Create output directory
    OUTPUT_DIR="output/${MODEL_DIR}_dialogues/${PROFILE_TYPE}"
    
    echo "========================================================"
    echo "Running batch for model: $MODEL"
    echo "Using profiles from: $PROFILE_DIR"
    echo "Output directory: $OUTPUT_DIR"
    echo "========================================================"
    
    # Create output directory if it doesn't exist
    mkdir -p "$OUTPUT_DIR"
    
    # Process all profiles in the directory
    PROFILES_TO_PROCESS=()
    for PROFILE_FILE in "$PROFILE_DIR"/*.json; do
      if [ ! -f "$PROFILE_FILE" ]; then
        continue
      fi
      
      # Extract profile filename
      PROFILE_FILENAME=$(basename "$PROFILE_FILE")
      
      # Determine output filename
      RAG_INDICATOR="without_rag"
      OUTPUT_FILENAME="${PROFILE_FILENAME//_user_/_dialogue_${NUM_TURNS}turns_${RAG_INDICATOR}_}"
      OUTPUT_PATH="$OUTPUT_DIR/$OUTPUT_FILENAME"
      
      PROFILES_TO_PROCESS+=("$PROFILE_FILE")
      # echo "Will process: $PROFILE_FILENAME (Output: $OUTPUT_FILENAME)"
    done
    
    echo "Found ${#PROFILES_TO_PROCESS[@]} profiles to process"
    
    # User prompt for continuing or skipping if not already set to run all
    if [ "$RUN_ALL" = false ]; then
      echo "Options:"
      echo "  1: Skip this batch"
      echo "  2: Run this batch"
      echo "  3: Run all remaining batches without prompting"
      read -p "Enter your choice (1-3): " CHOICE
      
      case $CHOICE in
        1)
          echo "Skipping this batch..."
          continue
          ;;
        3)
          echo "Running all remaining batches without further prompts..."
          RUN_ALL=true
          ;;
        *)
          echo "Running this batch..."
          ;;
      esac
    fi
    
    # Check if it's an OpenAI model
    if [[ "$MODEL" == openai/* ]]; then
      echo "OpenAI model detected. Using API key from .env file..."
      
      # Run with OpenAI API key from .env
      python src/batch_dialogue_generation.py \
        --max-concurrent "$MAX_CONCURRENT" \
        --assistant-model "$MODEL" \
        --num-turns "$NUM_TURNS" \
        --dialogues-dir "$OUTPUT_DIR" \
        --api-key "$OPENAI_API_KEY"
    else
      # Run the batch dialogue generation normally for non-OpenAI models
      echo "Starting batch processing for $MODEL..."
      
      python src/batch_dialogue_generation.py \
        --max-concurrent "$MAX_CONCURRENT" \
        --assistant-model "$MODEL" \
        --num-turns "$NUM_TURNS" \
        --dialogues-dir "$OUTPUT_DIR"
    fi
    
    # Check exit status
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
      # Increment failure counter for this model
      MODEL_FAILURES=$((MODEL_FAILURES + 1))
      echo "❌ Batch for $MODEL failed with exit code $EXIT_CODE (failure $MODEL_FAILURES/$MAX_FAILURES)"
    else
      echo "✅ Successfully processed $MODEL with $PROFILE_TYPE"
    fi
    
    # Add a separator between runs
    echo ""
    echo "Completed $MODEL with $PROFILE_TYPE"
    echo "----------------------------------------"
    echo ""
  done
done

echo "All batch dialogue generation jobs completed!" 