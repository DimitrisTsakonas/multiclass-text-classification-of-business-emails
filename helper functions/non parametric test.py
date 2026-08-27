"""
Purpose:
    Runs a paired Wilcoxon signed-rank test comparing macro-F1 scores across
    folds between two embedding variants.

Thesis reference:
    Section 5.6.2, Experiment 6 (Embeddings Variations - Token pooling).
    Compares sentence-averaged vs. token-pooled BERT embeddings; reproduces
    the reported W=0, p=0.0010 result.

Inputs:
    Two hardcoded 11-element arrays of per-fold macro-F1 scores, pasted in
    by hand rather than read from a file.

Outputs:
    Printed statistic/p-value/conclusion only.
"""
import numpy as np
from scipy import stats


def perform_wilcoxon_test(scores_model1: np.ndarray, scores_model2: np.ndarray, alpha: float = 0.05):
    """
    Performs a Wilcoxon signed-rank test to compare two paired sets of scores.

    Args:
        scores_model1 (np.ndarray): An array of scores for the first model.
                                     Each score corresponds to a specific fold.
        scores_model2 (np.ndarray): An array of scores for the second model.
                                     Each score corresponds to the same fold as in scores_model1.
        alpha (float): The significance level for the test (default is 0.05).

    Returns:
        tuple: A tuple containing:
            - statistic (float): The Wilcoxon test statistic.
            - p_value (float): The two-sided p-value.
            - conclusion (str): A string indicating the statistical conclusion.
    """
    if len(scores_model1) != len(scores_model2):
        raise ValueError("The number of scores for both models must be the same for a paired test.")

    statistic, p_value = stats.wilcoxon(scores_model1, scores_model2, alternative='two-sided')

    if p_value < alpha:
        conclusion = f"There is a statistically significant difference (p < {alpha})."
    else:
        conclusion = f"No statistically significant difference (p >= {alpha})."

    return statistic, p_value, conclusion

# Your 11 macro F1 scores for each model, corresponding to each test fold.
sentence_f1_scores = np.array([0.73,	0.77,	0.8,	0.8,	0.83,	0.81,	0.8,	0.82,	0.78,	0.76,	0.77])
word_f1_scores = np.array([0.77,	    0.83,	0.82,	0.85,	0.84,	0.86,	0.85,	0.85,	0.81,	0.8,	0.81])


# Perform the test
wilcoxon_stat, wilcoxon_p_value, test_conclusion = perform_wilcoxon_test(sentence_f1_scores, word_f1_scores)

print(f"Wilcoxon Statistic: {wilcoxon_stat:.4f}")
print(f"P-value:            {wilcoxon_p_value:.4f}")
print(f"Conclusion: {test_conclusion}")