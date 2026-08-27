"""
Purpose:
    Grid of per-class F1 score bar charts across the 11 train-test folds
    (one subplot per class), using preprocessing Set-D.

Thesis reference:
    Section 5.1, Experiment 1 (Initial Exploration), Figure 5.4.

Inputs:
    Hardcoded literal data (pasted from experiment output), not read from a
    file.

Outputs:
    Saves exp1_grid.pdf to the current working directory and shows the
    figure.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

# Folds
folds = np.arange(1, 12)

# Your data array (already rounded, 12 classes x 11 folds)
data = np.array([
    [0.00, 0.10, 0.85, 0.78, 0.85, 0.87, 0.86, 0.87, 0.78, 0.76, 0.88],  # Class 0
    [0.94, 0.95, 0.95, 0.95, 0.94, 0.94, 0.94, 0.95, 0.95, 0.95, 0.94],  # Class 1
    [0.43, 0.50, 0.46, 0.46, 0.46, 0.50, 0.48, 0.45, 0.47, 0.38, 0.48],  # Class 2
    [0.31, 0.67, 0.94, 0.50, 0.61, 0.69, 0.74, 0.75, 1.00, 0.00, 0.50],  # Class 3
    [0.94, 0.94, 0.96, 0.97, 0.95, 0.92, 0.96, 0.95, 0.96, 0.94, 0.95],  # Class 4
    [0.88, 0.92, 0.92, 0.93, 0.91, 0.95, 0.94, 0.92, 0.95, 0.94, 0.93],  # Class 5
    [0.97, 0.98, 0.98, 0.98, 0.98, 0.98, 0.97, 0.97, 0.98, 0.98, 0.98],  # Class 6
    [0.89, 0.91, 0.93, 0.91, 0.93, 0.93, 0.94, 0.93, 0.93, 0.91, 0.88],  # Class 7
    [0.93, 0.96, 0.91, 0.95, 0.97, 0.95, 0.87, 0.95, 0.98, 0.98, 0.93],  # Class 8
    [0.82, 0.86, 0.71, 0.80, 0.78, 0.84, 0.85, 0.81, 0.73, 0.59, 0.83],  # Class 9
    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],  # Class 10
    [0.61, 0.58, 0.69, 0.73, 0.79, 0.67, 0.68, 0.75, 0.79, 0.77, 0.63],  # Class 11
])

# Class labels
class_labels = [f"Class {i}" for i in range(12)]

# --- Compact layout settings ---
fig, axes = plt.subplots(4, 3, figsize=(10, 7))
axes = axes.flatten()

for i, ax in enumerate(axes):
    ax.bar(folds, data[i], color='#4C72B0', width=0.8)
    ax.set_title(class_labels[i], fontsize=11, pad=3, weight="bold")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.tick_params(axis='both', which='major', labelsize=9, length=3)
    ax.set_xticks(range(1, 12))
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # Hide redundant labels for cleaner grid
    if i % 3 != 0:
        ax.set_ylabel('')
    else:
        ax.set_ylabel('F1', fontsize=10)
    if i < 9:
        ax.set_xlabel('')
    else:
        ax.set_xlabel('Fold', fontsize=10)

# Adjust layout and save
plt.suptitle('Class-wise F1 Scores across folds', fontsize=13, y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=1.0, w_pad=0.6)
plt.savefig('exp1_grid.pdf', bbox_inches='tight', dpi=300)
plt.show()
