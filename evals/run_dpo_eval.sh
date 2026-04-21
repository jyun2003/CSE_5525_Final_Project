#!/bin/bash
#SBATCH --account=PAS3272
#SBATCH --gpus-per-node=1
#SBATCH --time=16:00:00
#SBATCH --mem=32GB
#SBATCH --cluster=ascend

# Set environment variables
export CC=gcc
export CXX=g++
export TRITON_CACHE_DIR=/fs/scratch/PAS3272/${USER}/triton_cache
export UV_CACHE_DIR=/fs/scratch/PAS3272/${USER}/.cache/uv
export OPENAI_API_KEY="sk-dummy-not-used"
export HF_TOKEN="YOUR HUGGING FACE TOKEN"
export VLLM_ENABLE_V1_MULTIPROCESSING=0

# Ensure uv is installed
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

# Activate the virtual environment
source .venv/bin/activate

# Setup safety-eval (only if needed)
cd /fs/scratch/PAS3272/roblero7/CSE_5525_Final_Project/oe_eval/dependencies/safety
if [ ! -d "safety-eval" ]; then
    bash install.sh
fi

# Set model path
model=/fs/scratch/PAS3272/roblero7/CSE_5525_Final_Project/checkpoints/dpo_merged_001

# Define dataset names
dataset_name=(
    "gsm8k"
    "mbpp"
    "ifeval"
    "harmbench::default"
    "xstest::default"
)

# Loop over datasets and run evaluations
for dataset in "${dataset_name[@]}"; do
    echo "Evaluating on ${dataset}..."
    if [ "${dataset}" == "gsm8k" ]; then
        num_shots=8
    else
        num_shots=0
    fi
    olmes \
        --model ${model} \
        --model-args '{"chat_model": true}' \
        --task ${dataset} \
        --num-shots ${num_shots} \
        --output-dir /fs/scratch/PAS3272/roblero7/CSE_5525_Final_Project/results/dpo_testing/${dataset}
done