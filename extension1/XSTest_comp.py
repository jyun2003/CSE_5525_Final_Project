from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


categories = [
    "Correct\nCompliance",
    "Over-\nRefusal",
    "Correct\nRefusal",
    "Under-Refusal\n(Benign)",
    "Under-Refusal\n(Harmful)"
]

# Best SFT percentages
sft = [50.0, 12.0, 38.0, 0.0, 0.0]

# DPO beta = 0.01 percentages
dpo = [62.0, 2.0, 22.0, 12.0, 2.0]


output_dir = Path("figures")
output_dir.mkdir(parents=True, exist_ok=True)


x = np.arange(len(categories))
width = 0.36

fig, ax = plt.subplots(figsize=(11, 6))

bars1 = ax.bar(x - width/2, sft, width, label="Best SFT")
bars2 = ax.bar(x + width/2, dpo, width, label="DPO (β = 0.01)")

# Labels and title
ax.set_title("XSTest Failure Type Distribution: Best SFT vs DPO (β = 0.01)", pad=18)
ax.set_ylabel("Percentage (%)")
ax.set_xlabel("Failure Type")
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylim(0, 70)

# Legend moved to upper right
ax.legend(loc="upper right")

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10
        )

plt.tight_layout()

# Save figure
png_path = output_dir / "xstest_sft_vs_dpo_barchart.png"
pdf_path = output_dir / "xstest_sft_vs_dpo_barchart.pdf"

plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
plt.show()

print(f"Saved to: {png_path}")
print(f"Saved to: {pdf_path}")