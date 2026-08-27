"""
Purpose:
    Compares the misclassifications made by the TF-IDF model against the BERT
    token-pooled embeddings model, to determine how many TF-IDF errors were
    correctly classified when using embeddings instead.

Thesis reference:
    Experiment 8 - Combined Representation (Section 5.8). Reproduces the
    analysis behind Table 5.22 and Figure 5.11, including the reported
    finding that 32.1% of TF-IDF misclassifications were unique to TF-IDF
    (i.e. correctly classified by embeddings).

Inputs:
    - <path-to-results>/59_misclassified_data.pkl  (TF-IDF run)
    - <path-to-results>/60_misclassified_data.pkl  (BERT run)
    Each .pkl is a dict keyed by month, containing indices, true labels, and
    predictions for the misclassified instances of that fold.

Outputs:
    Printed per-month breakdown to the console (not saved to a file).
"""

import pickle
from collections import Counter

tfidf_path = r"<path-to-results>/59_misclassified_data.pkl"
bert_path = r"<path-to-results>/60_misclassified_data.pkl"

with open(tfidf_path, 'rb') as file:
    tfidf_errors = pickle.load(file)

print("breakpoint")
with open(bert_path, 'rb') as file:
    bert_errors = pickle.load(file)

only_in_tfidf_true_labels = {}

for month in tfidf_errors.keys():
    if month in bert_errors:
        idx_tfidf = set(tfidf_errors[month].get("idx", []))
        idx_bert = set(bert_errors[month].get("idx", []))


        common = idx_tfidf & idx_bert  # Intersection: common elements
        only_in_tfidf = idx_tfidf - idx_bert  # Elements unique to dict_a
        only_in_bert = idx_bert - idx_tfidf  # Elements unique to dict_b

        percentage =round(len(only_in_tfidf)*100/len(idx_tfidf),1 )

        # Retrieve true labels for the unique TF-IDF misclassifications
        true_labels = tfidf_errors[month].get('true_label', [])
        unique_true_labels = [true_labels[i] for i, idx in enumerate(tfidf_errors[month].get('idx', [])) if
                              idx in only_in_tfidf]
        all_labels = range(11)
        label_counts = {label: 0 for label in all_labels}
        # Count occurrences of each true label
        label_counts.update(Counter(unique_true_labels))
        # Store the true labels in the dictionary
        only_in_tfidf_true_labels[month] = label_counts

        print(f"Month: {month}")
        print(f"total tfidf/bert errors:  {len(idx_tfidf)} / {len(idx_bert)}")
        print(f"out of which common:      {len(common)}")
        print(f"unique to tfidf:          {len(only_in_tfidf)}                 (i.e bert did those correct)")
        print(f"tfidf + bert improvement: {percentage}% ")
        print(f"labels of unique tfidf errors: (i.e instances that bert did correct:")
        print(f"{label_counts}")
        print(f" {label_counts[0]} {label_counts[1]} {label_counts[2]} {label_counts[3]} {label_counts[4]} {label_counts[5]} {label_counts[6]} {label_counts[7]} {label_counts[8]} {label_counts[9]} {label_counts[10]}")

        # print(f"Unique to bert_errors: {only_in_bert}")
        print()  # Blank line for readability

print("end")