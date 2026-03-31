"""
Merge LoRA adapter into base model and add Llama-3.2-1B-Instruct chat template.

Usage examples:
    python merge_llama_instruct.py --adapter checkpoints/sft_lora --output checkpoints/sft_merged_instruct
    python merge_llama_instruct.py --adapter checkpoints/dpo_lora --output checkpoints/dpo_merged
"""
import argparse
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
def merge_llama_instruct(base_model: str, adapter_path: str, output_path: str):
    print(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(base_model)
 
    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
 
    print("Merging adapter into base model...")
    model = model.merge_and_unload()
 
    print(f"Saving merged model to: {output_path}")
    model.save_pretrained(output_path)
 
    # Use the Instruct tokenizer — same vocab, but includes the correct chat template and stop token configuration
    print("Saving tokenizer from: meta-llama/Llama-3.2-1B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    tokenizer.save_pretrained(output_path)
    
    # Patch chat_template into tokenizer_config.json
    # (some transformers versions don't persist it on save)
    config_path = os.path.join(output_path, "tokenizer_config.json")
    with open(config_path) as f:
        config = json.load(f)
 
    if "chat_template" not in config:
        print("Patching chat_template into tokenizer_config.json...")
        config["chat_template"] = tokenizer.chat_template
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
            
    print("Done — merged model with Instruct tokenizer saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="meta-llama/Llama-3.2-1B", help="Base model name")
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter checkpoint")
    parser.add_argument("--output", required=True, help="Path to save merged model")
    args = parser.parse_args()

    merge_llama_instruct(args.base, args.adapter, args.output)