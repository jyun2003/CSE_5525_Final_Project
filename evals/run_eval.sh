#!/bin/bash
#SBATCH --account=PAS3272
#SBATCH --cluster=ascend
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=32GB

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

source /fs/scratch/PAS3272/huang4978/.venv/bin/activate

cd evals/olmes/oe_eval/dependencies/safety
bash install.sh
cd /fs/scratch/PAS3272/huang4978/CSE_5525_Final_Project

dataset_name=(
    "gsm8k"
    "mbpp"
    "ifeval"
    "harmbench::wildguard_reasoning_answer"
)

model_path=meta-llama/Llama-3.2-1B

for dataset in "${dataset_name[@]}"; do
    echo "Evaluating on ${dataset}..."
    olmes \
        --model ${model_path} \
        --task ${dataset} \
        --output-dir results/base/${dataset}
done