# STORM: Advanced Dialogue Generation System

STORM is a sophisticated dialogue generation system that implements asymmetric conversation generation with Retrieval-Augmented Generation (RAG) capabilities. The system is designed to create natural, context-aware dialogues between users and assistants.

## Core Features

- Asymmetric dialogue generation using separate LLM models
- RAG (Retrieval-Augmented Generation) support
- Emotional state and intent tracking
- Inner thoughts generation
- Comprehensive dialogue analysis and reporting
- Batch processing capabilities
- Profile-based dialogue generation

## Project Structure

```
storm/
├── src/                    # Core source code
├── scripts/               # Utility scripts
│   ├── setup.sh          # Environment setup script
│   ├── run_demo.sh       # Demo dialogue generation
│   ├── run_convert.sh    # Data conversion utilities
│   ├── run_profile_generate.sh    # Profile generation
│   ├── run_turn_analysis.sh      # Turn-by-turn analysis
│   ├── import_and_test_rag.sh    # RAG system setup and testing
│   └── run_batch_dialogues.sh    # Batch dialogue generation
├── example_data/         # Sample data and configurations
├── tests/               # Test suite
└── requirements.txt     # Python dependencies
```

If you want to run it, please move the `example_data/profiles` folder to the root directory first.

## Quick Start

1. Clone the repository:
   ```bash
   cd storm
   ```

2. Run the setup script:
   ```bash
   ./scripts/setup.sh
   ```

3. Set your API keys:
   ```bash
   export OPENAI_API_KEY='your_openai_api_key'
   export OPENROUTER_API_KEY='your_openrouter_api_key'
   ```

## Scripts Overview

### Setup and Configuration
- `setup.sh`: Initializes the environment and installs dependencies
- `run_profile_generate.sh`: Generates user profiles for dialogue generation

### Core Functionality
- `run_demo.sh`: Runs a single dialogue generation demo
- `run_batch_dialogues.sh`: Processes multiple dialogues in batch mode
- `run_turn_analysis.sh`: Analyzes dialogue turns and generates reports

### RAG System
- `import_and_test_rag.sh`: Sets up and tests the RAG system
- `run_convert.sh`: Converts and processes dialogue data

## Usage Examples

### Running a Demo Dialogue
```bash
./scripts/run_demo.sh 
```

### Batch Processing
```bash
./scripts/run_batch_dialogues.sh 
```

### RAG System Setup
```bash
./scripts/import_and_test_rag.sh 
```


## License

This project is licensed under the MIT License - see the LICENSE file for details. 
