from datasets import load_dataset
import json

# Load dataset
dataset = load_dataset(
    "allenai/tulu-3-sft-olmo-2-mixture-0225",
    split="train"
)

# Stats
total = 0
kept = 0

# Track duplicates
seen = set()

def is_good(example):
    messages = example["messages"]

    # Must have at least one user + assistant
    if len(messages) < 2:
        return False

    # Ensure last message is assistant
    if messages[-1]["role"] != "assistant":
        return False

    user_text = messages[0]["content"]
    assistant_text = messages[-1]["content"]

    # -------- LENGTH FILTER --------
    if len(user_text.split()) < 5:
        return False

    if len(assistant_text.split()) < 20:
        return False

    if len(assistant_text.split()) > 1200:
        return False

    # -------- DUPLICATE FILTER --------
    key = (user_text.strip(), assistant_text.strip())
    if key in seen:
        return False
    seen.add(key)

   

    return True


output_path = "data/tulu3_filtered.jsonl"

with open(output_path, "w") as f:
    for ex in dataset:
        total += 1

        if is_good(ex):
            json.dump({"messages": ex["messages"]}, f)
            f.write("\n")
            kept += 1

# -------- PRINT STATS --------
print("\n===== FILTERING STATS =====")
print(f"Total examples: {total}")
print(f"Kept examples:  {kept}")
print(f"Removed:        {total - kept}")
print(f"Kept %:         {kept / total * 100:.2f}%")
print("===========================\n")