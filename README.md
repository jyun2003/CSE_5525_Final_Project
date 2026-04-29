# CSE 5525: Default Project - Spring 2026

## Overview

This is the default project for CSE 5525 (Foundations of Speech and Language Processing). In this project, you will implement and train a language model using various training paradigms, then evaluate its performance on several benchmark tasks.

## Project Structure

```
├── README.md                 # This file
├── train_sft.py              # Template for Supervised Fine-Tuning
├── train_rm.py               # Template for Reward Modeling
├── train_pref.py             # Template for Preference Optimization
├── configs/                  # Configuration files for training
├── scripts/                  # Utility scripts
└── evals/                    # Evaluation suite (OLMES)
    ├── run_eval.sh           # Script to run evaluations
    └── olmes/                # AI2's Open Language Model Evaluation System
```

## Getting Started

### 1. Environment Setup

Set up your Python environment with the required dependencies:

```bash
# Clone the repository
git clone --recurse-submodules https://github.com/jyun2003/CSE_5525_Final_Project.git
cd CSE_5525_Final_Project

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies (adjust based on your requirements)
pip install tinker uv
git clone https://github.com/thinking-machines-lab/tinker-cookbook.git
cd tinker-cookbook
pip install -e .

# For data analysis and filtering (Extension 2)
pip install datasets pandas datasketch
```

### 2. Training
#### Supervised Fine-Tuning (SFT)
```bash
cd CSE_5525_Final_Project
python train_sft.py --config configs/sft_baseline.yaml

# Download final sampler weights
tinker checkpoint download $TINKER_SAMPLER_PATH
# rename the downloaded lora adapter to a better name like sft_lora and move it to checkpoints directory

# merge the downloaded lora adapter into baseline meta-llama/Llama-3.2-1B
python merge_chat.py --adapter checkpoints/sft_lora --output checkpoints/sft_merged
```

#### Preference Optimization (PREF: DPO)
```bash
cd CSE_5525_Final_Project
python train_pref.py --config configs/dpo_01.yaml

tinker checkpoint download $TINKER_SAMPLER_PATH
python merge_chat.py --adapter checkpoints/dpo_lora --output checkpoints/dpo_merged
```

#### Extension 2:

##### Filtered SFT
```bash
python filter_tulu.py
python train_sft.py --config configs/sft_filtered.yaml

tinker checkpoint download $TINKER_SAMPLER_PATH
python merge_chat.py --adapter checkpoints/sft_filtered_lora --output checkpoints/sft_filtered
```

##### Filtered DPO (beta = 0.1)
```bash
python train_pref.py --config configs/dpo_filtered_01.yaml

tinker checkpoint download $TINKER_SAMPLER_PATH
python merge_chat.py --adapter checkpoints/dpo_filtered_lora --output checkpoints/dpo_filtered
```

##### Filtered DPO (beta = 0.01)
```bash
python train_pref.py --config configs/dpo_filtered_001.yaml

tinker checkpoint download $TINKER_SAMPLER_PATH
python merge_chat.py --adapter checkpoints/dpo_filtered_001_lora --output checkpoints/dpo_filtered_001
```
## Evaluation

### Evaluation Tasks
Your model will be evaluated on the following benchmarks:

| Task | Description |
|------|-------------|
| **GSM8K** | Grade school math word problems (mathematical reasoning) |
| **IFEval** | Instruction following evaluation |
| **MBPP** | Mostly Basic Python Problems (code generation) |
| **HarmBench** | Safety and harmfulness evaluation |
| **XSTest** | Safety and harmfulness evaluation |

### Running Evaluations
#### Evaluating SFT
```bash
# modify based on your directory structure
model path: CSE_5525_Final_Project/checkpoints/sft_role_final

then run:
sbatch evals/run_sft_eval.sh
```
#### Evaluating DPO
```bash
# modify based on your directory structure
model path: CSE_5525_Final_Project/checkpoints/dpo_merged_01_1100

then run:
sbatch evals/dpo_01_eval.sh
```

#### For Monitoring:
```bash
# Monitor with:
squeue -u <username>
# See the output file (named slurm-<jobid>.out)
tail -f slurm-<jobid>.out
```