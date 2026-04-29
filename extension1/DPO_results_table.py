from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# Data
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


output_dir = Path("figures")
output_dir.mkdir(parents=True, exist_ok=True)


fig, ax = plt.subplots(figsize=(13, 2.6))
ax.axis("off")

# Format table text
cell_text = []
for _, row in df.iterrows():
    cell_text.append([
        row["Model"],
        f'{row["Beta"]:.2f}'.rstrip("0").rstrip("."),
        f'{row["GSM8K"]:.2f}',
        f'{row["MBPP"]:.2f}',
        f'{row["IFEval"]:.2f}',
        f'{row["HarmBench"]:.2f}',
        f'{row["XSTest"]:.2f}',
        f'{row["Avg. Task Perf."]:.2f}',
    ])

col_labels = list(df.columns)

# Draw table
table = ax.table(
    cellText=cell_text,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
    colLoc="center",
)


table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.15, 1.9)

# Header style
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight="bold")
        cell.set_facecolor("#EAEAEA")
    else:
        cell.set_facecolor("white")
    cell.set_edgecolor("lightgray")
    cell.set_linewidth(0.8)

# Left-align model names
for r in range(1, len(df) + 1):
    table[(r, 0)].get_text().set_ha("left")


plt.tight_layout()
fig.savefig(output_dir / "dpo_results_table.png", dpi=300, bbox_inches="tight")
fig.savefig(output_dir / "dpo_results_table.pdf", bbox_inches="tight")
plt.close(fig)

print("Done. Table saved to:", output_dir.resolve())