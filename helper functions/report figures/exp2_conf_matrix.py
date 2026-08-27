"""
Purpose:
    Renders a confusion matrix as a seaborn heatmap (12 classes, before
    class 2 removal), capped at the 85th percentile value for color scale
    readability.

Thesis reference:
    Section 5.2.2.1, Experiment 2 (Method Evaluation - Class 2 removal),
    Figure 5.6 - confusion matrix before the removal of class 2.

Inputs:
    Hardcoded literal matrix (pasted from experiment output), not read from
    a file.

Outputs:
    Saves exp2_conf_matrix_exp29.pdf under Chapter_4/.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === 1. Paste your matrix here ===
cm = np.array([
    [26, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 19051, 478, 0, 1, 1, 343, 0, 2, 74, 0, 1],
    [0, 649, 527, 0, 0, 0, 4, 13, 0, 0, 0, 0],
    [0, 0, 0, 17, 1, 0, 0, 0, 0, 0, 0, 0],
    [4, 44, 0, 0, 3427, 111, 25, 13, 0, 0, 0, 5],
    [0, 35, 0, 0, 74, 2085, 167, 1, 0, 0, 0, 2],
    [0, 342, 0, 0, 0, 4, 21161, 0, 12, 1, 0, 1],
    [0, 2, 0, 0, 3, 0, 0, 329, 0, 0, 0, 2],
    [0, 142, 0, 0, 0, 0, 100, 3, 1685, 0, 0, 13],
    [0, 13, 0, 0, 0, 0, 10, 0, 19, 87, 0, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 0],
    [0, 71, 0, 0, 17, 3, 34, 0, 15, 1, 0, 274]
])


# Compute the 85th percentile
percentile_85 = np.percentile(cm, 85)
print(f"85th percentile value: {percentile_85:,.0f}")

# === 2. Define labels (0–10 classes) ===
labels = [str(i) for i in range(12)]

# === 3. Define output path ===
output_dir = r"<path-to-repo>\helper functions\report figures\Chapter_4"
os.makedirs(output_dir, exist_ok=True)
out_path = os.path.join(output_dir, "exp2_conf_matrix_exp29.pdf")

# === 4. Plot ===
plt.figure(figsize=(10, 8))
ax = sns.heatmap(
    cm,
    annot=True,         # Set True if you want numbers on top of colors
    fmt='d',
    cmap='Greens',       # “Google Sheets–style” gradient
    cbar=False,
    xticklabels=labels,
    yticklabels=labels,
    vmax=percentile_85,
    linewidths=0.5,  # grid between cells
    linecolor='gray',  # grid color
    square=True,
)



# === 6. Label styling ===
plt.xlabel("Predicted Class", fontsize=12, fontweight='bold', labelpad=10)
plt.ylabel("True Class", fontsize=12, fontweight='bold', labelpad=10)

# === 7. Add “Predicted Class” on top as well ===
# Duplicate x-label at top
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.set_xlabel("Predicted Class", fontsize=12, fontweight='bold', labelpad=10)

# === 8. Adjust tick labels ===
plt.xticks(rotation=0, fontsize=9)
plt.yticks(rotation=0, fontsize=9)

plt.tight_layout()
plt.savefig(out_path, bbox_inches="tight")
plt.close()

print(f"Confusion matrix saved to:\n{out_path}")
