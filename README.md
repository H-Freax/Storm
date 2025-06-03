# STORM: Official Repository for "WHEN TO ACT, WHEN TO WAIT"

> Structured Task-Oriented Representation Model for Intent Triggerability  
> in Task-Oriented Dialogue Systems

Welcome to the official repository for **STORM**, a framework that models asymmetric information dynamics between user and agent LLMs to track and trigger user intents effectively in task-oriented dialogue.


<p align="center">

<a href="https://nanostorm.netlify.app/" target="_blank" rel="noopener noreferrer" style="margin: 6px;">
  <img src="https://img.shields.io/badge/Project-Page-1E90FF?style=flat-square&logo=chromium&logoColor=white" alt="Project Page" />
</a>

<a href="#" target="_blank" rel="noopener noreferrer" style="margin: 6px;">
  <img src="https://img.shields.io/badge/arXiv-PDF--Coming--Soon-DC143C?style=flat-square&logo=arxiv&logoColor=white" alt="arXiv PDF Coming Soon" />
</a>

<a href="https://github.com/H-Freax/Storm" target="_blank" rel="noopener noreferrer" style="margin: 6px;">
  <img src="https://img.shields.io/badge/GitHub-Code-181717?style=flat-square&logo=github&logoColor=white" alt="GitHub Code" />
</a>

<a href="https://huggingface.co/datasets/FreaxRuby/storm" target="_blank" rel="noopener noreferrer" style="margin: 6px;">
  <img src="https://img.shields.io/badge/Dataset-HuggingFace-FF6F00?style=flat-square&logo=huggingface&logoColor=white" alt="HuggingFace Dataset" />
</a>

<a href="https://v0-dialogue-analysis-dashboard.vercel.app/" target="_blank" rel="noopener noreferrer" style="margin: 6px;">
  <img src="https://img.shields.io/badge/Dashboard-Visualization-0066CC?style=flat-square&logo=chartdotjs&logoColor=white" alt="Dashboard Visualization" />
</a>

</p>


---

<h2 align="center">Authors</h2>

<p align="center">

<a href="https://h-freax.github.io/" target="_blank" rel="noopener noreferrer" style="margin: 4px;">
  <img src="https://img.shields.io/badge/Yaoyao%20Qian%20(Lead)-Northeastern%20University-0366d6?style=for-the-badge&logo=university&logoColor=white" alt="Yaoyao Qian (Lead) — Northeastern University" />
</a>

<a href="https://jindanh.github.io/" target="_blank" rel="noopener noreferrer" style="margin: 4px;">
  <img src="https://img.shields.io/badge/Jindan%20Huang-Tufts%20University-254473?style=for-the-badge&logo=university&logoColor=white" alt="Jindan Huang — Tufts University" />
</a>

<a href="https://pentium3.github.io/" target="_blank" rel="noopener noreferrer" style="margin: 4px;">
  <img src="https://img.shields.io/badge/Yuanli%20Wang-Boston%20University-cc0000?style=for-the-badge&logo=university&logoColor=white" alt="Yuanli Wang — Boston University" />
</a>

<a href="https://simonucl.github.io/" target="_blank" rel="noopener noreferrer" style="margin: 4px;">
  <img src="https://img.shields.io/badge/Simon%20Yu-Northeastern%20University-0366d6?style=for-the-badge&logo=university&logoColor=white" alt="Simon Yu — Northeastern University" />
</a>

<a href="https://kyriezz.com/" target="_blank" rel="noopener noreferrer" style="margin: 4px;">
  <img src="https://img.shields.io/badge/Kyrie%20Zhixuan%20Zhou-UT%20San%20Antonio-004c97?style=for-the-badge&logo=university&logoColor=white" alt="Kyrie Zhixuan Zhou — UT San Antonio" />
</a>

<a href="https://jiayuanm.com/" target="_blank" rel="noopener noreferrer" style="margin: 4px;">
  <img src="https://img.shields.io/badge/Jiayuan%20Mao-MIT-a31f34?style=for-the-badge&logo=academic-cap&logoColor=white" alt="Jiayuan Mao — MIT" />
</a>

<a href="https://www.mingfuliang.com/" target="_blank" rel="noopener noreferrer" style="margin: 4px;">
  <img src="https://img.shields.io/badge/Mingfu%20Liang-Northwestern%20University-4e2a84?style=for-the-badge&logo=university&logoColor=white" alt="Mingfu Liang — Northwestern University" />
</a>

<a href="https://hanhanzhou.com/" target="_blank" rel="noopener noreferrer" style="margin: 4px;">
  <img src="https://img.shields.io/badge/Hanhan%20Zhou-George%20Washington%20University-5a2d81?style=for-the-badge&logo=university&logoColor=white" alt="Hanhan Zhou — George Washington University" />
</a>

</p>

---

<p align="center">
  <strong>Overview of the STORM Framework</strong>
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
