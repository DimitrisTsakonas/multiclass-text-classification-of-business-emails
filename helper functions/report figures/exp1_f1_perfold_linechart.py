"""
Purpose:
    Line chart of macro F1 score across the 11 train-test folds, using
    preprocessing Set-D.

Thesis reference:
    Section 5.1, Experiment 1 (Initial Exploration), Figure 5.3.

Inputs:
    Hardcoded literal data (pasted from experiment output), not read from a
    file.

Outputs:
    Saves exp1_f1_perfold_linechart.png to the current working directory
    and shows the figure.
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
# Data
folds = list(range(1, 12))
macro_f1 = [0.73, 0.78, 0.86, 0.83, 0.85, 0.85, 0.85, 0.86, 0.88, 0.77, 0.83]

# Plot
plt.figure(figsize=(7.5, 4.5))
plt.plot(folds, macro_f1, marker='o', linewidth=2, color='#4C72B0', label='Macro F1')

# Labels / ticks / grid
plt.xlabel('Train-test folds', fontsize=11)
plt.ylabel('Macro F1', fontsize=11)
plt.xticks(range(1, 12, 1))
plt.grid(True, linestyle='--', alpha=0.4)
plt.ylim(0.7, 0.9)  # tighten range for readability
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

for x, y in zip(folds, macro_f1):
    plt.text(x, y + 0.01, f'{y:.2f}', ha='center', va='bottom', fontsize=9, color='#333333')


plt.tight_layout()
plt.savefig('exp1_f1_perfold_linechart.png', dpi=300, bbox_inches='tight')
plt.show()
