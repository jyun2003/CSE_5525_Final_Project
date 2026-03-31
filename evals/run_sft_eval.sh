#!/bin/bash
#SBATCH --account=PAS3272
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=32GB
 
export CC=gcc
export CXX=g++
export TRITON_CACHE_DIR=/fs/scratch/PAS3272/${USER}/triton_cache
export UV_CACHE_DIR=/fs/scratch/PAS3272/${USER}/.cache/uv
export OPENAI_API_KEY="sk-dummy-not-used"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
 
source /fs/scratch/PAS3272/huang4978/.venv/bin/activate
# cd evals/olmes/oe_eval/dependencies/safety
# bash install.sh
cd /fs/scratch/PAS3272/huang4978/CSE_5525_Final_Project
 
dataset_name=(
    "gsm8k"
    "mbpp"
    "ifeval"
)
 
model=/fs/scratch/PAS3272/huang4978/CSE_5525_Final_Project/checkpoints/sft_role_final
for dataset in "${dataset_name[@]}"; do
    echo "Evaluating SFT on ${dataset}..."
    olmes \
        --model ${model} \
        --model-args '{"chat_model": true}' \
        --task ${dataset} \
        --output-dir ./results/sft_role_final/${dataset}
done
 