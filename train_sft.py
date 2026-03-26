"""
This module implements the SFTTrainer class for training your model using supervised fine-tuning (SFT).

Usage:
    python train_sft.py --config configs/sft_baseline.yaml
    python train_sft.py --config configs/sft_smoke_test.yaml
    python train_sft.py --config configs/sft_filtered.yaml       # Extension 2
    python train_sft.py --config configs/sft_lora_sweep.yaml     # Extension 3
"""
import argparse
import asyncio
import yaml
from datetime import datetime

from tinker_cookbook import checkpoint_utils
from tinker_cookbook.supervised import train as sft_train
from tinker_cookbook.recipes.chat_sl.chat_datasets import Tulu3Builder
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
from tinker_cookbook.recipes.chat_sl.train import get_infrequent_evaluator_builders


# Configuration management
def load_config(config_path: str) -> dict:
    """
    Load training configuration from a YAML file in the configs/ directory

    Args:
        config_path: Path to a YAML config file (i.e. "configs/sft_baseline.yaml")

    Returns:
        Dict of training arguments.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print(f"Loaded config from: {config_path}")
    return config


# Data loading and preprocessing
def get_dataset_builder(dataset, model_name, renderer_name, max_length, batch_size):
    """
    Build a dataset loader based on the dataset name.

    Supports:
        - "tulu3"       : Tulu 3 SFT mixture (866K examples) — used for baseline
        - "*.jsonl"     : Custom JSONL file (used for Extension 2: filtered data)
    """
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=max_length,
        batch_size=batch_size,
    )

    if dataset == "tulu3":
        return Tulu3Builder(common_config=common_config)
    elif dataset.endswith(".jsonl"):
        return FromConversationFileBuilder(
            common_config=common_config,
            file_path=dataset,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# SFT Trainer
class SFTTrainer:
    def __init__(self, training_args=None):
        """
        Args:
            training_args: Dict with hyperparameters. Load from configs/ with load_config().
        """
        self.training_args = training_args or {}

    def train(self):
        # Read hyperparameters from config (with defaults)
        args = self.training_args
        model_name = args.get("model_name", "meta-llama/Llama-3.2-1B")
        dataset = args.get("dataset", "tulu3")
        learning_rate = args.get("learning_rate", 5e-4)
        lr_schedule = args.get("lr_schedule", "linear")
        num_epochs = args.get("num_epochs", 1)
        lora_rank = args.get("lora_rank", 64)
        batch_size = args.get("batch_size", 128)
        max_length = args.get("max_length", 16384)
        save_every = args.get("save_every", 500) # checkpoints are saved every `save_every` steps to `log_path`
        eval_every = args.get("eval_every", 500)
        infrequent_eval_every = args.get("infrequent_eval_every", 1000)
        max_steps = args.get("max_steps", None)
        wandb_project = args.get("wandb_project", "cse5525_sft")
        wandb_name = args.get("wandb_name", None)
        log_path = args.get("log_path", None)
        load_checkpoint_path = args.get("load_checkpoint_path", None) # training can be resumed by setting `load_checkpoint_path`
        renderer_name = args.get("renderer_name", None)
        inline_evals = args.get("inline_evals", None)
        base_url = args.get("base_url", None)

        # Generate run name and log path
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        run_name = f"SFT-{lora_rank}rank-{learning_rate}lr-{batch_size}batch-{timestamp}"
        if log_path is None:
            log_path = f"/fs/scratch/PAS3272/huang4978/CSE_5525_Final_Project/logs/{run_name}"
        if wandb_name is None:
            wandb_name = run_name

        renderer_name = checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default(
            model_name=model_name,
            explicit_renderer_name=renderer_name,
            load_checkpoint_path=load_checkpoint_path,
            base_url=base_url,
        )

        # Build the dataset
        dataset_builder = get_dataset_builder(
            dataset=dataset,
            model_name=model_name,
            renderer_name=renderer_name,
            max_length=max_length,
            batch_size=batch_size,
        )

        # inline evaluators (optional)
        infrequent_evaluator_builders = get_infrequent_evaluator_builders(
            inline_evals=inline_evals,
            renderer_name=renderer_name,
            model_name=model_name,
        )

        # Assemble the training config
        config = sft_train.Config(
            log_path=log_path,
            model_name=model_name,
            renderer_name=renderer_name,
            load_checkpoint_path=load_checkpoint_path,
            dataset_builder=dataset_builder,
            evaluator_builders=[],
            infrequent_evaluator_builders=infrequent_evaluator_builders,
            learning_rate=learning_rate,
            lr_schedule=lr_schedule,
            num_epochs=num_epochs,
            base_url=base_url,
            wandb_project=wandb_project,
            wandb_name=wandb_name,
            lora_rank=lora_rank,
            save_every=save_every,
            eval_every=eval_every,
            infrequent_eval_every=infrequent_eval_every,
            max_steps=max_steps,
        )

        # Run training
        print("=" * 60)
        print("Starting SFT Training")
        print("=" * 60)
        print(f"  Model:         {model_name}")
        print(f"  Dataset:       {dataset}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  LoRA rank:     {lora_rank}")
        print(f"  Batch size:    {batch_size}")
        print(f"  Epochs:        {num_epochs}")
        print(f"  Save every:    {save_every} steps")
        print(f"  Log path:      {log_path}")
        print(f"  W&B project:   {wandb_project}")
        if max_steps:
            print(f"  Max steps:     {max_steps} (smoke test)")
        print("=" * 60)

        asyncio.run(sft_train.main(config))


# CLI entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT Training for CSE 5525")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file (e.g., configs/sft_baseline.yaml)")

    cli_args = parser.parse_args()

    # Load config from YAML file
    training_args = load_config(cli_args.config)

    trainer = SFTTrainer(training_args=training_args)
    trainer.train()