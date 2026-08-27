"""
Purpose:
    Pie chart of TF-IDF misclassification overlap with embeddings: 14,478
    total misclassifications, split into 9,837 common with embeddings and
    4,641 unique to TF-IDF (i.e. correctly classified by embeddings).

Thesis reference:
    Section 5.8, Experiment 8 (Combined Representation), Figure 5.11. Ties
    to helper functions/missclassifications_handler.py, which computes the
    underlying counts. Note: despite the filename, this script renders a
    pie chart, not a Venn diagram - an earlier Venn-based version was
    replaced but the file was never renamed to match.

Inputs:
    Hardcoded literal counts (pasted from experiment output), not read from
    a file.

Outputs:
    Saves tfidf_pie_chart.png to the current working directory and shows
    the figure.
"""
from matplotlib import pyplot as plt

labels = [
    "Common errors with embeddings",
    "Correct using embeddings"
]

values = [9837, 4641]
colors = ['lightgrey', '#8FBC8F']

# Function to show percentage + absolute value

def autopct_format(values):
    def my_format(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        return f'{pct:.1f}%\n({val})'
    return my_format

# Plot
plt.figure(figsize=(7,7))

plt.pie(
    values,
    labels=labels,
    colors=colors,
    autopct=autopct_format(values),
    startangle=90,
    textprops={'fontsize': 14}   # controls slice text size
)

plt.axis('equal')

title = plt.title("Total TF-IDF Misclassifications (14,478)")
title.set_fontsize(16)

plt.savefig("tfidf_pie_chart.png", dpi=300, bbox_inches="tight")
plt.show()

