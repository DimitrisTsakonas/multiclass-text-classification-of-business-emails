"""
Purpose:
    Renders a confusion matrix as a seaborn heatmap (11 classes), capped at
    the 85th percentile value for color scale readability.

Thesis reference:
    Section 5.3, Experiment 3 (Full Scale Evaluation), Figure 5.8 -
    Logistic Regression's confusion matrix.

Inputs:
    Hardcoded literal matrix (pasted from experiment output), not read from
    a file.

Outputs:
    Saves exp3_conf_matrix_exp51.pdf under Chapter_4/.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === 1. Paste your matrix here ===
cm = np.array([
    [544,    0,    0,   36,    2,    0,    0,    0,    0,    0,    7],
    [0, 228053,    1,   65,  557, 3868,  255,  344,  303,    0,  350],
    [0,   14,  179,   19,    3,   10,    0,    0,    0,    0,   10],
    [82,  282,   30, 41898, 1995,  151,  110,    0,    0,    0,  175],
    [0,  676,    8, 1027, 46735, 1026,    3,    0,    1,    0,   84],
    [0, 3271,    5,   49,  208, 250259,    8,  343,   79,    0,  291],
    [0,   52,    0,    4,    1,    4,  4276,    9,    1,    0,   43],
    [0,  726,    0,    0,    2,  766,   15, 22352,   87,    0,  266],
    [0,   73,    0,    1,    0,   29,    0,   80, 1341,    0,   55],
    [0,    0,    0,    0,    0,    0,    0,    0,    0,  334,    0],
    [6,  365,    8,  155,   37,  152,   57,   96,   27,    0, 3573]
])


# Compute the 85th percentile
percentile_85 = np.percentile(cm, 85)
print(f"85th percentile value: {percentile_85:,.0f}")


# === 2. Define labels (0–10 classes) ===
labels = [str(i) for i in range(11)]

# === 3. Define output path ===
output_dir = r"<path-to-repo>\helper functions\report figures\Chapter_4"
os.makedirs(output_dir, exist_ok=True)
out_path = os.path.join(output_dir, "exp3_conf_matrix_exp51.pdf")

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
