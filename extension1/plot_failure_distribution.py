from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


categories = [
    "Correct / acceptable",
    "Reasoning / incorrect answer",
    "Instruction / format failure",
    "Hallucination / off-target / other",
]

best_sft = [20.7, 37.3, 22.7, 19.3]
dpo_beta_001 = [30.0, 41.3, 19.3, 9.3]


output_dir = Path("figures")
output_dir.mkdir(exist_ok=True)


x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))

bars1 = ax.bar(x - width/2, best_sft, width, label="Best SFT")
bars2 = ax.bar(x + width/2, dpo_beta_001, width, label="DPO (β = 0.01)")

# Labels and title
ax.set_ylabel("Percentage (%)")
ax.set_xlabel("Failure Type")
ax.set_title("Overall Failure-Type Distribution: Best SFT vs DPO (β = 0.01)")
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=15, ha="right")
ax.legend(loc="upper right")
ax.set_ylim(0, 50)

# Value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.8,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9
        )

plt.tight_layout()

# Save
png_path = output_dir / "failure_distribution_sft_vs_dpo_beta001.png"
pdf_path = output_dir / "failure_distribution_sft_vs_dpo_beta001.pdf"

plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
plt.show()

print(f"Saved PNG to: {png_path}")
print(f"Saved PDF to: {pdf_path}")