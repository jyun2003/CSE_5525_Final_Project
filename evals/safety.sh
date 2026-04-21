#!/bin/bash
#SBATCH --account=PAS3272
#SBATCH --gpus-per-node=1
#SBATCH --time=10:00:00
#SBATCH --mem=32GB
#SBATCH --cluster=ascend

#This part is need for OSC users
export CC=gcc
export CXX=g++
export TRITON_CACHE_DIR=/fs/scratch/PAS3272/${USER}/triton_cache
export UV_CACHE_DIR=/fs/scratch/PAS3272/${USER}/.cache/uv 
export OPENAI_API_KEY="sk-dummy-not-used"
export HF_TOKEN="YOUR HUGGING FACE TOKEN"
export VLLM_ENABLE_V1_MULTIPROCESSING=0

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Build olmes venv (only if needed — skip if already exists)
cd /fs/scratch/PAS3272/roblero7/CSE_5525_Final_Project/evals/olmes
if [ ! -d ".venv" ]; then
    uv venv
    uv sync
    uv sync --group gpu
fi

# Activate AFTER venv exists
source .venv/bin/activate

# Setup safety-eval (only if needed)
cd oe_eval/dependencies/safety
if [ ! -d "safety-eval" ]; then
    bash install.sh
fi

cd /fs/scratch/PAS3272/roblero7/CSE_5525_Final_Project
model=/fs/scratch/PAS3272/roblero7/CSE_5525_Final_Project/checkpoints/dpo_merged_05
echo "Evaluating on harmbench::default..."
olmes \
  --model ${model} \
  --model-args '{"chat_model": true}' \
  --task "harmbench::default" \
  --output-dir ./dpo_merged_05_safety \
