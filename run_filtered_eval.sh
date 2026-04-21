#!/bin/bash
#SBATCH --account=PAS3272
#SBATCH --gpus-per-node=1
#SBATCH --time=16:00:00
#SBATCH --mem=32GB
#SBATCH --cluster=ascend

export CC=gcc
export CXX=g++
export TRITON_CACHE_DIR=/fs/scratch/PAS3272/${USER}/triton_cache
export UV_CACHE_DIR=/fs/scratch/PAS3272/${USER}/.cache/uv
export OPENAI_API_KEY="sk-dummy-not-used"
export VLLM_ENABLE_V1_MULTIPROCESSING=0

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

cd /users/PAS3272/roblero7/CSE_5525_Final_Project/evals/olmes

if [ ! -d ".venv" ]; then
    uv venv
    uv sync
    uv sync --group gpu
fi

source .venv/bin/activate

model=/users/PAS3272/roblero7/CSE_5525_Final_Project/checkpoints/filtered_merged

dataset_name=(
    "gsm8k"
    "mbpp"
    "ifeval"
    "harmbench::default"
    "xstest::default"
)

for dataset in "${dataset_name[@]}"; do
    echo "Evaluating on ${dataset}..."

    if [ "${dataset}" == "gsm8k" ]; then
        num_shots=8
    else
        num_shots=0
    fi

    olmes \
        --model "${model}" \
        --model-args '{"chat_model": true}' \
        --task "${dataset}" \
        --num-shots "${num_shots}" \
        --output-dir "/users/PAS3272/roblero7/CSE_5525_Final_Project/results/filtered_eval/${dataset}"
done
