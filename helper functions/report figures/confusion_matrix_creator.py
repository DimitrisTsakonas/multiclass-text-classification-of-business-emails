"""
Purpose:
    Renders a confusion matrix as a seaborn heatmap (11 classes), capped at
    the 85th percentile value for color scale readability.

Thesis reference:
    Section 5.3, Experiment 3 (Full Scale Evaluation), Figure 5.7 -
    XGBoost's confusion matrix.

Inputs:
    Hardcoded literal matrix (pasted from experiment output), not read from
    a file.

Outputs:
    Saves exp3_conf_matrix_exp44.pdf under Chapter_4/.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === 1. Paste your matrix here ===
cm = np.array([
    [530,    0,    0,   50,    0,    2,    0,    0,    0,    0,    7],
    [0, 229966,    0,   11,  149, 3376,    3,  188,   80,    0,   23],
    [0,    2,  174,   33,   18,    2,    0,    0,    0,    0,    6],
    [39,   83,    2, 42530, 1874,  116,   27,    1,    0,    0,   51],
    [0,  151,    1, 1063, 48238,   82,    3,    1,    0,    0,   21],
    [0, 3071,    0,   20,  109, 251070,    0,  178,   13,    0,   52],
    [0,  201,    0,   23,    5,   10, 4109,   10,    0,    0,   32],
    [0,  789,    0,    2,    3,  803,   14, 22419,   28,    0,  156],
    [0,  131,    0,    0,    0,   62,    0,   70, 1280,    0,   36],
    [0,    0,    0,    0,    0,    0,    0,    0,    0,  332,    2],
    [0,  438,    2,  186,   89,  229,   60,  181,    8,    0, 3283]
])


# Compute the 85th percentile
percentile_85 = np.percentile(cm, 85)
print(f"85th percentile value: {percentile_85:,.0f}")


# === 2. Define labels (0–10 classes) ===
labels = [str(i) for i in range(11)]

# === 3. Define output path ===
output_dir = r"<path-to-repo>\helper functions\report figures\Chapter_4"
os.makedirs(output_dir, exist_ok=True)
out_path = os.path.join(output_dir, "exp3_conf_matrix_exp44.pdf")

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
