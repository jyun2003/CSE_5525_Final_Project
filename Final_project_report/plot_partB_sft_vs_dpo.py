from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


benchmarks = ["GSM8K", "MBPP", "IFEval", "HarmBench", "XSTest"]

# Best SFT checkpoint = SFT Full
sft_scores = [26.3078, 24.0, 46.4, 82.2, 82.8889]

# DPO results
dpo_001_scores = [30.2502, 23.2, 62.9496, 69.3750, 76.2222]   # beta = 0.01
dpo_01_scores  = [28.8097, 25.0, 50.4796, 77.5000, 84.2222]   # beta = 0.1
dpo_05_scores  = [26.6111, 23.6, 46.7626, 80.3125, 85.5556]   # beta = 0.5


output_dir = Path("figures")
output_dir.mkdir(parents=True, exist_ok=True)


plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 16,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
})

x = np.arange(len(benchmarks))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 7))

# Colors
color_sft = "#9E9E9E"      
color_001 = "#BBDEFB"      
color_01  = "#64B5F6"      
color_05  = "#1E88E5"      
bars1 = ax.bar(x - 1.5 * width, sft_scores, width, label="Best SFT",
               color=color_sft, edgecolor="black", linewidth=0.6)
bars2 = ax.bar(x - 0.5 * width, dpo_001_scores, width, label="DPO β=0.01",
               color=color_001, edgecolor="black", linewidth=0.6)
bars3 = ax.bar(x + 0.5 * width, dpo_01_scores, width, label="DPO β=0.1",
               color=color_01, edgecolor="black", linewidth=0.6)
bars4 = ax.bar(x + 1.5 * width, dpo_05_scores, width, label="DPO β=0.5",
               color=color_05, edgecolor="black", linewidth=0.6)

# title
ax.set_xticks(x)
ax.set_xticklabels(benchmarks)
ax.set_ylabel("Score (%)")
ax.set_title("Best SFT vs. DPO Across Beta Values")
ax.set_ylim(0, 100)
ax.grid(axis="y", alpha=0.25, linestyle="--")
ax.set_axisbelow(True)

# Legend moved to upper-left inside the plot
ax.legend(
    loc="upper left",
    bbox_to_anchor=(0.01, 0.99),
    frameon=True,
    facecolor="white",
    edgecolor="lightgray",
    ncol=1
)

# Annotate bars
def annotate_bars(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.8,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

for bars in [bars1, bars2, bars3, bars4]:
    annotate_bars(bars)

fig.tight_layout()

# Save
fig.savefig(output_dir / "partB_sft_vs_dpo_all_betas_clean.pdf", bbox_inches="tight")
fig.savefig(output_dir / "partB_sft_vs_dpo_all_betas_clean.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("Done. Figure saved to:", output_dir.resolve())