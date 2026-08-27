"""
Purpose:
    Bar chart of final vocabulary size for each of the 4 preprocessing sets
    (A-D), using the full 11-month training set.

Thesis reference:
    Section 5.1, Experiment 1 (Initial Exploration), Figure 5.1.

Inputs:
    Hardcoded literal data (pasted from experiment output), not read from a
    file.

Outputs:
    Saves exp1_11month_vocab.png to the current working directory and shows
    the figure.
"""
import matplotlib.pyplot as plt

# Data
sets = ['Set - A', 'Set - B', 'Set - C', 'Set - D']
vocab_sizes = [889008, 888698, 880441, 130552]

# Create bar chart
plt.figure(figsize=(7, 4))
bars = plt.bar(sets, vocab_sizes, color=['#4C72B0', '#55A868', '#C44E52', '#8172B3'])

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 5000, f"{height:,}",
             ha='center', va='bottom', fontsize=10)

# Labels and title
plt.title('11 Month Vocabulary Size', fontsize=12)
plt.xlabel('Preprocessing Set', fontsize=11)
plt.ylabel('Vocabulary Size', fontsize=11)

# Tidy up layout
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Save or show
plt.savefig('exp1_11month_vocab.png', dpi=300, bbox_inches='tight')
plt.show()
