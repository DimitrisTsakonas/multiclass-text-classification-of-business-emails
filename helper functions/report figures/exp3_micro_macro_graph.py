"""
Purpose:
    Side-by-side line charts comparing macro vs. micro F1 across all 11
    folds, one panel per model (Logistic Regression, XGBoost).

Thesis reference:
    Section 5.3, Experiment 3 (Full Scale Evaluation), Figure 5.9.

Inputs:
    Hardcoded literal data (pasted from experiment output), not read from a
    file.

Outputs:
    Saves exp3_micro_macro_graph.pdf to the current working directory and
    shows the figure.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

# Data
folds = np.arange(1, 12)

log_reg_macro = [0.86, 0.90, 0.91, 0.92, 0.94, 0.90, 0.90, 0.91, 0.86, 0.82, 0.91]
log_reg_micro = [0.97] * 11

xgb_macro = [0.87, 0.94, 0.92, 0.93, 0.95, 0.93, 0.92, 0.94, 0.94, 0.85, 0.94]
xgb_micro = [0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.97, 0.97, 0.98, 0.98, 0.97]

# Create figure and axes (shared y-axis)
fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)

# Define consistent colors
macro_color = '#1f77b4'  # blue
micro_color = '#ff7f0e'  # orange

# Font sizes
title_size = 16
label_size = 14
tick_size = 12
legend_size = 12
value_label_size = 12  # Increased for macro line value labels

# --- Logistic Regression plot ---
axes[0].plot(folds, log_reg_macro, marker='o', color=macro_color, label='Macro F1', linewidth=2)
axes[0].plot(folds, log_reg_micro, marker='s', color=micro_color, label='Micro F1', linewidth=2)
axes[0].set_title('Logistic Regression', fontsize=title_size)
axes[0].set_xlabel('Fold', fontsize=label_size)
axes[0].set_ylabel('F1 Score', fontsize=label_size)
axes[0].set_xticks(folds)
axes[0].tick_params(axis='both', labelsize=tick_size)
axes[0].grid(True, linestyle='--', alpha=0.6)

# Add labels for macro F1 points (below the line)
for x, y in zip(folds, log_reg_macro):
    axes[0].text(x, y - 0.010, f'{y:.2f}', ha='center', va='top', fontsize=value_label_size)

# --- XGBoost plot ---
axes[1].plot(folds, xgb_macro, marker='o', color=macro_color, label='Macro F1', linewidth=2)
axes[1].plot(folds, xgb_micro, marker='s', color=micro_color, label='Micro F1', linewidth=2)
axes[1].set_title('XGBoost', fontsize=title_size)
axes[1].set_xlabel('Fold', fontsize=label_size)
axes[1].set_xticks(folds)
axes[1].tick_params(axis='both', labelsize=tick_size)
axes[1].grid(True, linestyle='--', alpha=0.6)

# Add labels for macro F1 points (below the line)
for x, y in zip(folds, xgb_macro):
    axes[1].text(x, y - 0.010, f'{y:.2f}', ha='center', va='top', fontsize=value_label_size)

# Same y-axis for both
axes[0].set_ylim(0.8, 1.0)
axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
axes[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

# Shared legend below both plots
fig.legend(['Macro F1', 'Micro F1'],
           loc='lower center', ncol=2, fontsize=legend_size, frameon=False)

# Adjust layout and save
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("exp3_micro_macro_graph.pdf", format="pdf", bbox_inches="tight")
plt.show()
