"""
Purpose:
    Approximates token counts using TfidfVectorizer's analyzer (word-level,
    stopword-filtered) instead of a real tokenizer, on a differently
    preprocessed (stemmed) dataset version, as a comparison point to
    bert_token_counter.py.

Thesis reference:
    Secondary/comparison analysis, not itself cited in the thesis text.
    This script's truncation percentages (7.75% at 512 tokens, 0.33% at
    2048) do not match the thesis's quoted 29.29%/3.00%, confirming that
    bert_token_counter.py (the real BERT tokenizer) is the actual source of
    the reported Figure 5.10 numbers.

Inputs:
    Monthly .pkl files (dict with a text_key column, default
    'To From CC Subject Body') in data_folder_path.

Outputs:
    Printed stats; if output_folder is given, also saves a stats .txt file,
    the raw per-email token counts as a .npy array, and a histogram .png.
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


def token_stats_with_tfidf_tokenizer(
    data_folder_path,
    text_key="To From CC Subject Body",
    stop_words="english",
    min_df=5,
    ngram_range=(1, 1),
    output_folder=None,
    bin_width=100,
    max_tokens=None,
):

    # Create TF-IDF vectorizer ONLY to reuse its analyzer
    vectorizer = TfidfVectorizer(
        stop_words=stop_words,
        min_df=min_df,
        ngram_range=ngram_range,
    )
    analyzer = vectorizer.build_analyzer()

    # Collect and sort .pkl files
    pkl_files = sorted(
        os.path.join(data_folder_path, f)
        for f in os.listdir(data_folder_path)
        if f.endswith(".pkl")
    )

    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in: {data_folder_path}")

    token_counts_list = []

    # Outer progress bar for files
    for file_idx, fp in enumerate(tqdm(pkl_files, desc="Processing PKL files"), start=1):

        with open(fp, "rb") as f:
            d = pickle.load(f)

        if text_key not in d:
            raise KeyError(f"Key '{text_key}' not found in {fp}. Keys: {list(d.keys())}")

        texts = d[text_key]

        # Inner progress bar for emails
        for text in tqdm(texts, desc=f"Emails in file {file_idx}/{len(pkl_files)}", leave=False):
            token_counts_list.append(len(analyzer(text)))

    token_counts = np.array(token_counts_list, dtype=np.int32)

    # =========================
    # BASIC STATISTICS
    # =========================

    stats = {
        "n_emails": int(token_counts.size),
        "min": int(token_counts.min()),
        "max": int(token_counts.max()),
        "mean": float(token_counts.mean()),
        "median": float(np.median(token_counts)),
        "std": float(token_counts.std()),
        "p90": float(np.percentile(token_counts, 90)),
        "p95": float(np.percentile(token_counts, 95)),
        "p99": float(np.percentile(token_counts, 99)),
        "zeros": int((token_counts == 0).sum()),
    }

    print("\n=== TOKEN STATS (TFIDF TOKENIZER) ===")
    for k, v in stats.items():
        print(f"{k}: {v}")

    # =========================
    # TRUNCATION ANALYSIS
    # =========================

    total_emails = token_counts.size

    # BERT (512)
    bert_limit = 512
    bert_truncated = (token_counts > bert_limit).sum()
    bert_percentage = (bert_truncated / total_emails) * 100

    # Nomic (2048)
    nomic_limit = 2048
    nomic_truncated = (token_counts > nomic_limit).sum()
    nomic_percentage = (nomic_truncated / total_emails) * 100

    print("\n=== TRUNCATION ANALYSIS ===")
    print(f"BERT limit (512 tokens):")
    print(f"  Emails truncated: {bert_truncated}")
    print(f"  Percentage: {bert_percentage:.2f}%")

    print(f"\nNomic limit (2048 tokens):")
    print(f"  Emails truncated: {nomic_truncated}")
    print(f"  Percentage: {nomic_percentage:.2f}%")

    # =========================
    # OUTLIER ANALYSIS (>3000)
    # =========================

    cap_value = 3000
    above_cap = (token_counts > cap_value).sum()
    above_cap_percentage = (above_cap / total_emails) * 100

    print(f"\n=== OUTLIER ANALYSIS (> {cap_value}) ===")
    print(f"Emails above {cap_value} tokens: {above_cap}")
    print(f"Percentage: {above_cap_percentage:.4f}%")

    # =========================
    # SAVE RESULTS
    # =========================

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

        # Save statistics
        stats_path = os.path.join(output_folder, "token_stats.txt")
        with open(stats_path, "w", encoding="utf-8") as f:
            f.write("=== TOKEN STATS (TFIDF TOKENIZER) ===\n")
            for k, v in stats.items():
                f.write(f"{k}: {v}\n")

            f.write("\n=== TRUNCATION ANALYSIS ===\n")
            f.write(f"BERT truncated (>512): {bert_truncated} ({bert_percentage:.2f}%)\n")
            f.write(f"Nomic truncated (>2048): {nomic_truncated} ({nomic_percentage:.2f}%)\n")

        # Save raw counts
        np.save(os.path.join(output_folder, "token_counts.npy"), token_counts)

        # =========================
        # HISTOGRAM
        # =========================

        if max_tokens is None:
            max_tokens = int(token_counts.max())

        bins = np.arange(0, max_tokens + bin_width + 1, bin_width)

        plt.figure(figsize=(10, 6))
        plt.hist(token_counts, bins=bins, edgecolor="black")
        plt.xlabel("Tokens per email (TFIDF tokenizer)")
        plt.ylabel("Number of emails")
        plt.title(f"Token Histogram (bin width = {bin_width})")
        plt.tight_layout()

        hist_path = os.path.join(output_folder, f"token_hist_bin{bin_width}.png")
        plt.savefig(hist_path)
        plt.close()

        print("\nSaved files:")
        print(f"  {stats_path}")
        print(f"  token_counts.npy")
        print(f"  {hist_path}")

    return token_counts, stats


# =========================
# RUN
# =========================

if __name__ == "__main__":

    token_stats_with_tfidf_tokenizer(
        data_folder_path="<path-to-dataset>/Version 4.2.4 stemmed c special out",
        output_folder="<path-to-repo>/helper functions/token_counter_using_tfidf_tokenizer",
        stop_words="english",
        min_df=5,
        ngram_range=(1, 1),
        bin_width=100,
        max_tokens=3000,   # optional: cap histogram to make it readable
    )