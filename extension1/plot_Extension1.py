from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


data = {
    "Model": [
        "dpo_merged_001",
        "dpo_merged_01",
        "dpo_merged_05",
    ],
    "Beta": [0.01, 0.1, 0.5],
    "GSM8K": [30.25, 28.81, 26.61],
    "MBPP": [23.20, 25.00, 23.60],
    "IFEval": [62.95, 50.48, 46.76],
    "HarmBench": [69.38, 77.50, 80.31],
    "XSTest": [76.22, 84.22, 85.56],
    "Avg. Task Perf.": [38.80, 34.76, 32.32],
}

df = pd.DataFrame(data)


df["Safty"] = (df["HarmBench"] + df["XSTest"]) / 2


output_dir = Path("figures")
output_dir.mkdir(parents=True, exist_ok=True)


# Create plot
fig, ax = plt.subplots(figsize=(10, 7))

ax.plot(
    df["Safty"],
    df["Avg. Task Perf."],
    marker="o",
    linewidth=2,
    markersize=8,
)

# Annotate each point with beta value
for _, row in df.iterrows():
    beta_text = f'β={row["Beta"]:.2f}'.rstrip("0").rstrip(".")
    ax.annotate(
        beta_text,
        (row["Safty"], row["Avg. Task Perf."]),
        textcoords="offset points",
        xytext=(8, 8),
        fontsize=12,
    )

# Labels and title
ax.set_xlabel("Safty (%)", fontsize=14)
ax.set_ylabel("Avg. Task Perf. (%)", fontsize=14)
ax.set_title("Safty vs Avg. Task Perf. across DPO beta values", fontsize=18)

# Grid
ax.grid(True, alpha=0.3)


plt.tight_layout()
fig.savefig(output_dir / "safty_vs_avg_task_perf.png", dpi=300, bbox_inches="tight")
fig.savefig(output_dir / "safty_vs_avg_task_perf.pdf", bbox_inches="tight")
plt.close(fig)

print("Done. Figure saved to:", output_dir.resolve())