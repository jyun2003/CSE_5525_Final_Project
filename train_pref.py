"""
This module implements the PREFTrainer class for training your model using preference optimization.

Usage:
    python train_pref.py --config configs/dpo_baseline.yaml
"""

import yaml
from datetime import datetime

from tinker_cookbook import checkpoint_utils
from tinker_cookbook.preference import train_dpo
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
from tinker_cookbook.preference.dpo_datasets import DPODatasetBuilderFromComparisons
import chz
import datasets
from typing import cast
from tinker_cookbook.preference.preference_datasets import ComparisonDatasetBuilder
from tinker_cookbook.preference.types import Comparison, LabeledComparison

# Configuration management
def load_config(config_path: str) -> dict:
    """
    Load training configuration from a YAML file in the configs/ directory.

    Args:
        config_path: Path to a YAML config file (e.g., "configs/dpo_baseline.yaml")

    Returns:
        Dict of training arguments.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print(f"Loaded config from: {config_path}")
    return config

# Data loading and preprocessing
@chz.chz
class OLMo2ComparisonBuilder(ComparisonDatasetBuilder):
    """
    OLMo 2 1B Preference Mix dataset builder.
    Dataset: https://huggingface.co/datasets/allenai/olmo-2-0425-1b-preference-mix

    378,301 examples. Each example has 'chosen' and 'rejected' as lists of
    message dicts with 'role' and 'content' keys.
    """

    test_size: int = 1024

    def get_train_and_test_datasets(self) -> tuple[datasets.Dataset, datasets.Dataset | None]:
        dataset = datasets.load_dataset(
            "allenai/olmo-2-0425-1b-preference-mix", split="train"
        )
        dataset = cast(datasets.Dataset, dataset)
        dataset = dataset.shuffle(seed=0)
        test_dataset = dataset.take(self.test_size)
        train_dataset = dataset.skip(self.test_size)
        return train_dataset, test_dataset

    def example_to_labeled_comparison(self, example: dict) -> LabeledComparison | None:
        chosen = example["chosen"]
        rejected = example["rejected"]

        if len(chosen) < 2 or len(rejected) < 2:
            return None

        if chosen[:-1] != rejected[:-1]:
            return None

        comparison = Comparison(
            prompt_conversation=chosen[:-1],
            completion_A=[chosen[-1]],
            completion_B=[rejected[-1]],
        )
        return LabeledComparison(comparison=comparison, label="A")

# PREF Trainer
class PREFTrainer:
    def __init__(self, training_args=None):
        """
        Args:
            training_args: Dict with hyperparameters. Load from configs/ with load_config().
                Required key:
                    - load_checkpoint_path: Tinker URI of your best SFT checkpoint
                See configs/dpo_baseline.yaml for all available options.
        """
        self.training_args = training_args or {}

    def train(self):
        """
        Run Direct Preference Optimization via the Tinker API.

        Metrics to watch during training:
            - dpo_loss: Should decrease
            - accuracy: Should increase toward 0.6-0.7+
            - margin: Should increase (chosen_reward - rejected_reward)
            - If accuracy is stuck near 0.5, stop the run to save credits.
        """
        args = self.training_args

        # Read hyperparameters from config (with defaults)
        model_name = args.get("model_name", "meta-llama/Llama-3.2-1B")
        learning_rate = args.get("learning_rate", 0.00001)  # 1e-5
        lr_schedule = args.get("lr_schedule", "linear")
        num_epochs = args.get("num_epochs", 1)
        dpo_beta = args.get("dpo_beta", 0.1)
        lora_rank = args.get("lora_rank", 16)
        batch_size = args.get("batch_size", 256)
        max_length = args.get("max_length", 8192)
        save_every = args.get("save_every", 200)
        eval_every = args.get("eval_every", 200)
        infrequent_eval_every = args.get("infrequent_eval_every", 500)
        max_steps = args.get("max_steps", None)
        wandb_project = args.get("wandb_project", "cse5525_dpo")
        wandb_name = args.get("wandb_name", None)
        log_path = args.get("log_path", None)
        load_checkpoint_path = args.get("load_checkpoint_path", None)
        renderer_name = args.get("renderer_name", None)
        base_url = args.get("base_url", None)

        # Validate that SFT checkpoint is provided
        if load_checkpoint_path is None:
            raise ValueError(
                "load_checkpoint_path is required for DPO training.\n"
                "Set it in configs/dpo_baseline.yaml to your SFT Tinker URI.\n"
                "Example: tinker://your-session-id/checkpoint-step-XXXX"
            )

        # Generate run name and log path
        model_short = model_name.replace("/", "-")
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        run_name = f"dpo-beta{dpo_beta}-{lora_rank}rank-{learning_rate}lr-{timestamp}"
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
        
        # Build olmo2 dataset
        common_config = ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer=model_name,
            renderer_name=renderer_name,
            max_length=max_length,
            batch_size=batch_size,
        )
        dataset_builder = DPODatasetBuilderFromComparisons(
            common_config=common_config,
            comparison_builder=OLMo2ComparisonBuilder(),
        )

        # Assemble the training config
        config = train_dpo.Config(
            log_path=log_path,
            model_name=model_name,
            renderer_name=renderer_name, 
            dataset_builder=dataset_builder,
            load_checkpoint_path=load_checkpoint_path,
            learning_rate=learning_rate,
            lr_schedule=lr_schedule,
            num_epochs=num_epochs,
            dpo_beta=dpo_beta,
            lora_rank=lora_rank,
            base_url=base_url,
            wandb_project=wandb_project,
            wandb_name=wandb_name,
            save_every=save_every,
            eval_every=eval_every,
            infrequent_eval_every=infrequent_eval_every,
            max_steps=max_steps,
        )

        # ── Step 6: Run training ──────────────────────────────────────────
        print("=" * 60)
        print("Starting DPO Training")
        print("=" * 60)
        print(f"  Model:           {model_name}")
        print(f"  SFT checkpoint:  {load_checkpoint_path}")
        print(f"  Renderer:        {renderer_name}")
        print(f"  DPO beta:        {dpo_beta}")
        print(f"  Learning rate:   {learning_rate}")
        print(f"  LoRA rank:       {lora_rank}")
        print(f"  Batch size:      {batch_size}")
        print(f"  Epochs:          {num_epochs}")
        print(f"  Save every:      {save_every} steps")
        print(f"  Log path:        {log_path}")
        print(f"  W&B project:     {wandb_project}")
        if max_steps:
            print(f"  Max steps:       {max_steps} (smoke test)")
        print("=" * 60)

        train_dpo.main(config)


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DPO Training")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config (e.g., configs/dpo_baseline.yaml)")

    cli_args = parser.parse_args()

    training_args = load_config(cli_args.config)

    trainer = PREFTrainer(training_args=training_args)
    trainer.train()