"""
Purpose:
    Line chart of vocabulary growth across the 11 training folds, for each
    of the 4 preprocessing sets (A-D).

Thesis reference:
    Section 5.1, Experiment 1 (Initial Exploration), Figure 5.2.

Inputs:
    Hardcoded literal data (pasted from experiment output), not read from a
    file.

Outputs:
    Saves exp1_vocab_linechart.png to the current working directory and
    shows the figure.
"""
import matplotlib.pyplot as plt

# Data
folds = list(range(1, 12))
set_a = [134147, 224815, 308235, 377749, 458746, 534269, 604738, 691895, 760026, 823861, 889008]
set_b = [133854, 224516, 307932, 377444, 458441, 533964, 604429, 691586, 759717, 823551, 888698]
set_c = [130476, 220358, 303023, 371970, 452495, 527487, 597640, 684425, 752118, 815600, 880441]
set_d = [41743, 56049, 67922, 76728, 86256, 94411, 101985, 109666, 117170, 123731, 130552]

colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3']
# Plot
plt.figure(figsize=(8, 5))
plt.plot(folds, set_a, marker='o', color=colors[0], label='Set A')
plt.plot(folds, set_b, marker='s', color=colors[1], label='Set B')
plt.plot(folds, set_c, marker='^', color=colors[2], label='Set C')
plt.plot(folds, set_d, marker='D', color=colors[3], label='Set D')

# Labels and title
plt.title('Vocabulary Growth Across Training Folds', fontsize=12)
plt.xlabel('Training-set size (Months)', fontsize=11)
plt.ylabel('Vocabulary Size', fontsize=11)
plt.legend(title='Preprocessing Set', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)

plt.xticks(range(1, 12, 1))
# Adjust layout and save
plt.tight_layout()
plt.savefig('exp1_vocab_linechart.png', dpi=300, bbox_inches='tight')
plt.show()
