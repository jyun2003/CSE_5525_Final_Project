"""
Merge LoRA adapter into base model and add chat template.

Usage:
    python merge_chat.py --adapter checkpoints/sft_lora --output checkpoints/sft_merged
    python merge_chat.py --adapter checkpoints/dpo_lora --output checkpoints/dpo_merged
"""
import argparse
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}User: {{ message['content'] }}\n{% endif %}"
    "{% if message['role'] == 'assistant' %}Assistant: {{ message['content'] }}\n{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}Assistant:{% endif %}"
)

def merge_chat(base_model: str, adapter_path: str, output_path: str):
    print(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(base_model)

    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("Merging adapter into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {output_path}")
    model.save_pretrained(output_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.save_pretrained(output_path)

    # Patch chat_template into tokenizer_config.json
    config_path = os.path.join(output_path, "tokenizer_config.json")
    with open(config_path) as f:
        config = json.load(f)
    config["chat_template"] = CHAT_TEMPLATE
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print("Done — merged model with chat template saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="meta-llama/Llama-3.2-1B", help="Base model name")
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter checkpoint")
    parser.add_argument("--output", required=True, help="Path to save merged model")
    args = parser.parse_args()

    merge_chat(args.base, args.adapter, args.output)