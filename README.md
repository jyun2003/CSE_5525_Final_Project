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

# For evaluation (OLMES)
cd evals/olmes
sbatch run_eval.sh

# Monitor with:
squeue -u roblero7
# See the output file (named slurm-<jobid>.out)
tail -f slurm-<jobid>.out
```

### 2. Training

We provide three template files for different training approaches:

#### Supervised Fine-Tuning (SFT)
Implement your SFT training logic in `train_sft.py`. This is the standard approach for instruction-tuning language models.

An example of how to do this has already been provider by Tinker for you [here](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/chat_sl)

### Training Supervised Fine-Tuning (SFT)
```bash
cd CSE_5525_Final_Project
python train_sft.py --config configs/sft_baseline.yaml

# Download final sampler weights
tinker checkpoint download $TINKER_SAMPLER_PATH
# rename the downloaded lora adapter to a better name like sft_lora and move it to checkpoints directory

# merge the downloaded lora adapter into baseline meta-llama/Llama-3.2-1B
python merge_chat.py --adapter checkpoints/sft_lora --output checkpoints/sft_merged
```
### Evaluating SFT
```bash
# modify based on your directory structure
model path: CSE_5525_Final_Project/checkpoints/sft_merged

then run:
sbatch evals/run_sft_eval.sh
```

### DPO
```bash
cd CSE_5525_Final_Project
python train_pref.py --config configs/dpo_01.yaml

tinker checkpoint download $TINKER_SAMPLER_PATH
python merge_chat.py --adapter checkpoints/dpo_lora --output checkpoints/dpo_merged
```
### Evaluating DPO
```bash
# modify based on your directory structure
model path: CSE_5525_Final_Project/checkpoints/dpo_merged

then run:
sbatch evals/run_dpo_eval.sh
```

#### Reward Modeling (RM)
Implement your reward model training in `train_rm.py`. This trains a model to predict human preferences.

An example of how to do this has already been provider by Tinker for you [here](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/preference/rlhf)

#### Preference Optimization (PREF)
Implement your preference optimization (e.g., DPO, PPO) in `train_pref.py`. This aligns the model using preference data.

An example of how to do this has already been provider by Tinker for you [here](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/preference)

Each template provides a basic class structure. You should:
1. Complete the `train()` method with your training loop
2. Add data loading and preprocessing
3. Implement checkpointing and logging
4. Add configuration management via the `configs/` directory

## Evaluation

After training your model, you must evaluate it using **OLMES** (Open Language Model Evaluation System), AI2's evaluation suite.

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

1. **Setup OLMES:**

```bash
cd evals/olmes

# Install with uv (recommended)
export CC=gcc
export CXX=g++
uv sync
uv sync --group gpu  # for GPU/vLLM support

# Or install with pip
pip install -e .
pip install -e ".[gpu]"  # for GPU support
```

2. **Run evaluations:**
You can decide to run your evaluations on Tinker or on OSC. If you decide to use Tinker for evaluation, you will be responsible to porting this code evaluation harness into Tinker for your usage.

To this code on OSC, please replace `xxxx` in `run_eval.sh` with the correct project.

Use the provided evaluation script:

```bash
cd ../
bash run_eval.sh
```

Or run individual evaluations:

```bash
# GSM8K (Mathematical Reasoning)
olmes --model <your-model-path> --task gsm8k --output-dir <output-dir>

# IFEval (Instruction Following)
olmes --model <your-model-path> --task ifeval --output-dir <output-dir>

# MBPP (Code Generation)
olmes --model <your-model-path> --task mbpp --output-dir <output-dir>

# HarmBench (Safety Evaluation)
olmes --model <your-model-path> --task harmbench::default --output-dir <output-dir>
```

### Evaluation Output

Each evaluation will produce:
- `predictions.jsonl` - Model predictions for each task instance
- `metrics.json` - Aggregated metrics and scores
- Detailed logs and analysis

## Submission Requirements

You must submit the following:

### 1. Prediction Files
Submit the `predictions.jsonl` file for **each** evaluation task:
- `gsm8k-predictions.jsonl`
- `ifeval-predictions.jsonl`
- `mbpp-predictions.jsonl`
- `harmbench-predictions.jsonl`

### 2. Final Model
Submit your trained model checkpoint (or provide a link if hosted on HuggingFace Hub).

### 3. Report
Submit a written report documenting:
- Your approach and methodology
- Training details (hyperparameters, data used, compute resources)
- Results and analysis
- Ablation studies (if any)
- Discussion of limitations and future work

## Grading Criteria

Your project will be graded on:
1. **Implementation Quality** - Clean, well-documented code
2. **Model Performance** - Scores on the evaluation benchmarks
3. **Report Quality** - Clarity, completeness, and analysis depth
4. **Innovation** - Creative approaches or improvements

## Tips

- Start with SFT as a baseline before trying more advanced methods
- Monitor training loss and validation metrics carefully
- Use smaller batch sizes with gradient accumulation if memory is limited
- Test your evaluation pipeline early with a small model
- Document your experiments thoroughly

## Resources

- [OLMES Documentation](https://github.com/allenai/olmes)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [TRL (Transformer Reinforcement Learning)](https://huggingface.co/docs/trl)

## Questions?

If you have questions about the project, please:
1. Post them on Teams/Carman
2. Attend office hours
3. Email the course TA (Abraham) or Instructor (Sachin)

Good luck!