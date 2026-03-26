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

# control your uv caches
export UV_CACHE_DIR=/fs/scratch/PAS3272/${USER}/.cache/uv 

# Dummy key to prevent import error in safety-eval (WildGuard doesn't actually use it)
export OPENAI_API_KEY="sk-dummy-not-used"

# Disable vLLM V1 multiprocessing so EngineCore runs inline in the spawned subprocess
# rather than forking a grandchild process that loses CUDA visibility on SLURM
export VLLM_ENABLE_V1_MULTIPROCESSING=0
# export VLLM_MAX_MODEL_LEN=4096
# export VLLM_GPU_MEMORY_UTILIZATION=0.85
# export VLLM_GPU_MEMORY_UTILIZATION=0.60
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Make uv available
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# source /fs/scratch/PAS3272/huang4978/.venv/bin/activate
source /fs/scratch/PAS3272/huang4978/CSE_5525_Final_Project/evals/olmes/.venv/bin/activate
cd /fs/scratch/PAS3272/huang4978/CSE_5525_Final_Project/evals/olmes
uv venv --clear
uv sync
uv sync --group gpu

cd oe_eval/dependencies/safety
rm -rf safety-eval
bash install.sh
# cd evals/olmes/oe_eval/dependencies/safety
# [ -d safety-eval ] || bash install.sh

# cd /fs/scratch/PAS3272/huang4978/CSE_5525_Final_Project/evals/olmes
# uv venv         # create UV venv if it doesn't exist
# uv run -- python -m ensurepip --upgrade  # ensure pip exists
# uv run -- python -m pip install fastchat

cd /fs/scratch/PAS3272/huang4978/CSE_5525_Final_Project

# model=meta-llama/Llama-3.2-1B-Instruct
model=/fs/scratch/PAS3272/huang4978/CSE_5525_Final_Project/checkpoints/sft_merged
echo "Evaluating on harmbench::wildguard_reasoning_answer..."

olmes \
  --model ${model} \
  --model-args '{"chat_model": true}' \
  --task "harmbench::wildguard_reasoning_answer" \
  --output-dir ./test \
