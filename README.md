# STORM: Official Repository for "WHEN TO ACT, WHEN TO WAIT"

> Structured Task-Oriented Representation Model for Intent Triggerability  
> in Task-Oriented Dialogue Systems

Welcome to the official repository for **STORM**, a framework that models asymmetric information dynamics between user and agent LLMs to track and trigger user intents effectively in task-oriented dialogue.

<p align="center" style="font-size:0;">

  <a href="https://arxiv.org/abs/XXXX.XXXXX" target="_blank" rel="noopener noreferrer" style="display:inline-block;">
    <img src="https://img.shields.io/badge/arXiv-PDF-A42C25?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv PDF" style="display:block; height:28px;"/>
  </a>&nbsp;

  <a href="https://nanostorm.netlify.app/" target="_blank" rel="noopener noreferrer" style="display:inline-block;">
    <img src="https://img.shields.io/badge/Project-Page-1E90FF?style=for-the-badge&logo=chromium&logoColor=white" alt="Project Page" style="display:block; height:28px;" />
  </a>&nbsp;

  <a href="https://github.com/H-Freax/Storm" target="_blank" rel="noopener noreferrer" style="display:inline-block;">
    <img src="https://img.shields.io/badge/GitHub-Code-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub Code" style="display:block; height:28px;" />
  </a>&nbsp;

  <a href="https://huggingface.co/datasets/FreaxRuby/storm" target="_blank" rel="noopener noreferrer" style="display:inline-block;">
    <img src="https://img.shields.io/badge/Dataset-HuggingFace-FF6F00?style=for-the-badge&logo=huggingface&logoColor=white" alt="HuggingFace Dataset" style="display:block; height:28px;" />
  </a>&nbsp;

  <a href="https://v0-dialogue-analysis-dashboard.vercel.app/" target="_blank" rel="noopener noreferrer" style="display:inline-block;">
    <img src="https://img.shields.io/badge/Dashboard-Visualization-0066CC?style=for-the-badge&logo=chartdotjs&logoColor=white" alt="Dashboard Visualization" style="display:block; height:28px;" />
  </a>

</p>




---
<h2 align="center">Authors</h2>
<p align="center" style="font-size:1.1em; line-height:1.6;">

  <span style="position: relative; display: inline-block; margin: 0 20px; padding-right: 20px; font-weight: 600;">
    <a href="https://h-freax.github.io/" target="_blank" rel="noopener noreferrer" style="text-decoration:none; color: inherit;">
      Yaoyao Qian 
    </a>
    <img src="https://upload.wikimedia.org/wikipedia/commons/b/bb/NU_RGB_seal_R.png" alt="Northeastern University" width="16" style="position: absolute; top: 0; right: 0;" />
  </span>&nbsp;&nbsp;&nbsp;

  <span style="position: relative; display: inline-block; margin: 0 20px; padding-right: 20px; font-weight: 600;">
    <a href="https://jindanh.github.io/" target="_blank" rel="noopener noreferrer" style="text-decoration:none; color: inherit;">
      Jindan Huang
    </a>
    <img src="https://upload.wikimedia.org/wikipedia/en/thumb/b/b1/Tufts_official_seal.svg/1920px-Tufts_official_seal.svg.png" alt="Tufts University" width="16" style="position: absolute; top: 0; right: 0;" />
  </span>&nbsp;&nbsp;&nbsp;

  <span style="position: relative; display: inline-block; margin: 0 20px; padding-right: 20px; font-weight: 600;">
    <a href="https://pentium3.github.io/" target="_blank" rel="noopener noreferrer" style="text-decoration:none; color: inherit;">
      Yuanli Wang
    </a>
    <img src="https://upload.wikimedia.org/wikipedia/en/thumb/1/15/Boston_University_Terriers_logo.svg/1200px-Boston_University_Terriers_logo.svg.png" alt="Boston University" width="16" style="position: absolute; top: 0; right: 0;" />
  </span>&nbsp;&nbsp;&nbsp;

  <span style="position: relative; display: inline-block; margin: 0 20px; padding-right: 20px; font-weight: 600;">
    <a href="https://simonucl.github.io/" target="_blank" rel="noopener noreferrer" style="text-decoration:none; color: inherit;">
      Simon Yu
    </a>
    <img src="https://upload.wikimedia.org/wikipedia/commons/b/bb/NU_RGB_seal_R.png" alt="Northeastern University" width="16" style="position: absolute; top: 0; right: 0;" />
  </span>

  <br />

  <span style="position: relative; display: inline-block; margin: 0 20px; padding-right: 20px; font-weight: 600;">
    <a href="https://kyriezz.com/" target="_blank" rel="noopener noreferrer" style="text-decoration:none; color: inherit;">
      Kyrie Zhixuan Zhou
    </a>
    <img src="https://upload.wikimedia.org/wikipedia/en/thumb/8/8b/University_of_Texas_at_San_Antonio_seal.svg/1200px-University_of_Texas_at_San_Antonio_seal.svg.png" alt="UT San Antonio" width="16" style="position: absolute; top: 0; right: 0;" />
  </span>&nbsp;&nbsp;&nbsp;

  <span style="position: relative; display: inline-block; margin: 0 20px; padding-right: 20px; font-weight: 600;">
    <a href="https://jiayuanm.com/" target="_blank" rel="noopener noreferrer" style="text-decoration:none; color: inherit;">
      Jiayuan Mao
    </a>
    <img src="https://upload.wikimedia.org/wikipedia/commons/0/0c/MIT_logo.svg" alt="MIT" width="16" style="position: absolute; top: 0; right: 0;" />
  </span>&nbsp;&nbsp;&nbsp;

  <span style="position: relative; display: inline-block; margin: 0 20px; padding-right: 20px; font-weight: 600;">
    <a href="https://www.mingfuliang.com/" target="_blank" rel="noopener noreferrer" style="text-decoration:none; color: inherit;">
      Mingfu Liang
    </a>
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Northwestern_University_seal.svg/1200px-Northwestern_University_seal.svg.png" alt="Northwestern University" width="16" style="position: absolute; top: 0; right: 0;" />
  </span>&nbsp;&nbsp;&nbsp;

  <span style="position: relative; display: inline-block; margin: 0 20px; padding-right: 20px; font-weight: 600;">
    <a href="https://hanhanzhou.com/" target="_blank" rel="noopener noreferrer" style="text-decoration:none; color: inherit;">
      Hanhan Zhou
    </a>
    <img src="https://upload.wikimedia.org/wikipedia/en/thumb/d/d8/George_Washington_University_seal.svg/1920px-George_Washington_University_seal.svg.png" alt="George Washington University" width="16" style="position: absolute; top: 0; right: 0;" />
  </span>

</p>



<p align="center" style="margin-top: 1.2em; line-height: 1.6; font-size: 1em;">
  <span style="vertical-align: middle; margin-right: 6px;">
    <img src="https://upload.wikimedia.org/wikipedia/commons/b/bb/NU_RGB_seal_R.png" alt="Northeastern University" width="20" style="position: relative; top: 2px;">
  </span> Northeastern University &nbsp;&nbsp;&nbsp;
  
  <span style="vertical-align: middle; margin-right: 6px;">
    <img src="https://upload.wikimedia.org/wikipedia/en/thumb/b/b1/Tufts_official_seal.svg/1920px-Tufts_official_seal.svg.png" alt="Tufts University" width="20" style="position: relative; top: 2px;">
  </span> Tufts University &nbsp;&nbsp;&nbsp;
  
  <span style="vertical-align: middle; margin-right: 6px;">
    <img src="https://upload.wikimedia.org/wikipedia/en/thumb/1/15/Boston_University_Terriers_logo.svg/1200px-Boston_University_Terriers_logo.svg.png" alt="Boston University" width="20" style="position: relative; top: 2px;">
  </span> Boston University &nbsp;&nbsp;&nbsp;
  
  <span style="vertical-align: middle; margin-right: 6px;">
    <img src="https://upload.wikimedia.org/wikipedia/en/thumb/8/8b/University_of_Texas_at_San_Antonio_seal.svg/1200px-University_of_Texas_at_San_Antonio_seal.svg.png" alt="UT San Antonio" width="20" style="position: relative; top: 2px;">
  </span> University of Texas at San Antonio
  <br />
  <span style="vertical-align: middle; margin-right: 6px;">
    <img src="https://upload.wikimedia.org/wikipedia/commons/0/0c/MIT_logo.svg" alt="MIT" width="20" style="position: relative; top: 2px;">
  </span> Massachusetts Institute of Technology &nbsp;&nbsp;&nbsp;
  
  <span style="vertical-align: middle; margin-right: 6px;">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Northwestern_University_seal.svg/1200px-Northwestern_University_seal.svg.png" alt="Northwestern University" width="20" style="position: relative; top: 2px;">
  </span> Northwestern University &nbsp;&nbsp;&nbsp;
  
  <span style="vertical-align: middle; margin-right: 6px;">
    <img src="https://upload.wikimedia.org/wikipedia/en/thumb/d/d8/George_Washington_University_seal.svg/1920px-George_Washington_University_seal.svg.png" alt="George Washington University" width="20" style="position: relative; top: 2px;">
  </span> George Washington University
</p>



<p align="center" style="margin-top: 2em; font-size: 0.9em; color: #555;">
  <span style="vertical-align: middle;">&#128231;</span>
  &nbsp;<strong>Corresponding author:</strong> 
  <a href="https://h-freax.github.io/" target="_blank" rel="noopener noreferrer" style="color: #0366d6; text-decoration: none; font-weight: 600;">
    Yaoyao Qian
  </a>
</p>




---

<p align="center">
  <h1>Overview of the STORM Framework</h1>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/a456f5ec-cc82-4a8b-a1c1-5569d8674df3" alt="STORM Architecture" />
</p>

---

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




## 📬 Contact

For questions, bug reports, or collaboration inquiries, please contact:  
**Yaoyao(Freax) Qian** — [qian.ya@northeastern.edu](mailto:qian.ya@northeastern.edu)


## License

This project is licensed under the MIT License - see the LICENSE file for details. 
