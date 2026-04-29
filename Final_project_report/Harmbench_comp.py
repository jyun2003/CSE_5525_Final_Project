from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


categories = [
    "Safe refusal",
    "Non-refusal safe answer",
    "Unsafe compliance"
]

sft = [70, 12, 18]
dpo_001 = [16, 56, 28]


x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(x - width/2, sft, width, label="Best SFT")
bars2 = ax.bar(x + width/2, dpo_001, width, label="DPO (β = 0.01)")


ax.set_title("HarmBench Output Type Distribution: Best SFT vs DPO (β = 0.01)")
ax.set_ylabel("Percentage (%)")
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=15)
ax.set_ylim(0, 100)

# Put legend in upper left
ax.legend(loc="upper left")


def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.0f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom")

add_labels(bars1)
add_labels(bars2)

plt.tight_layout()


output_dir = Path("figures")
output_dir.mkdir(exist_ok=True)

output_path = output_dir / "harmbench_sft_vs_dpo_barchart.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved to: {output_path}")