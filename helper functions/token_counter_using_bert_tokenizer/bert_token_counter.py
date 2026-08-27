"""
Purpose:
    Computes untruncated token-count statistics for the dataset using the
    real BERT tokenizer, plus a truncation-percentage analysis at the
    512-token (BERT) and 2048-token (Nomic) context limits.

Thesis reference:
    Section 5.7, Experiment 7 (Nomic Embeddings) - very likely the source of
    Figure 5.10 ("Distribution of token counts per email in the dataset").
    This script's saved output (29.29% of emails exceed 512 tokens, 3.00%
    exceed 2048) matches the thesis's quoted truncation percentages exactly.

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
from tqdm import tqdm
from transformers import AutoTokenizer


def token_stats_with_bert_tokenizer(
    data_folder_path,
    text_key="To From CC Subject Body",
    output_folder=None,
    bin_width=100,
    max_tokens=None,
    model_name="bert-base-uncased",
):

    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Collect pkl files
    pkl_files = sorted(
        os.path.join(data_folder_path, f)
        for f in os.listdir(data_folder_path)
        if f.endswith(".pkl")
    )

    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in: {data_folder_path}")

    token_counts_list = []

    # Process files
    for file_idx, fp in enumerate(tqdm(pkl_files, desc="Processing PKL files"), start=1):

        with open(fp, "rb") as f:
            d = pickle.load(f)

        texts = d[text_key]

        for text in tqdm(texts, desc=f"Emails in file {file_idx}/{len(pkl_files)}", leave=False):
            # IMPORTANT: no truncation here — we want full length
            tokens = tokenizer(
                text,
                add_special_tokens=True,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )

            token_counts_list.append(len(tokens["input_ids"]))

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
    }

    print("\n=== TOKEN STATS (BERT TOKENIZER) ===")
    for k, v in stats.items():
        print(f"{k}: {v}")

    # =========================
    # TRUNCATION ANALYSIS
    # =========================

    total_emails = token_counts.size

    bert_limit = 512
    bert_truncated = (token_counts > bert_limit).sum()
    bert_percentage = (bert_truncated / total_emails) * 100

    nomic_limit = 2048
    nomic_truncated = (token_counts > nomic_limit).sum()
    nomic_percentage = (nomic_truncated / total_emails) * 100

    print("\n=== TRUNCATION ANALYSIS (BERT TOKENIZER) ===")
    print(f"BERT limit (512 tokens):")
    print(f"  Emails truncated: {bert_truncated}")
    print(f"  Percentage: {bert_percentage:.2f}%")

    print(f"\n2048 token limit:")
    print(f"  Emails above 2048: {nomic_truncated}")
    print(f"  Percentage: {nomic_percentage:.2f}%")

    # =========================
    # SAVE + HISTOGRAM
    # =========================

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

        stats_path = os.path.join(output_folder, "bert_token_stats.txt")
        with open(stats_path, "w", encoding="utf-8") as f:
            f.write("=== TOKEN STATS (BERT TOKENIZER) ===\n")
            for k, v in stats.items():
                f.write(f"{k}: {v}\n")

            f.write("\n=== TRUNCATION ANALYSIS ===\n")
            f.write(f"BERT >512: {bert_truncated} ({bert_percentage:.2f}%)\n")
            f.write(f">2048: {nomic_truncated} ({nomic_percentage:.2f}%)\n")

        np.save(os.path.join(output_folder, "bert_token_counts.npy"), token_counts)

        if max_tokens is None:
            max_tokens = int(token_counts.max())

        bins = np.arange(0, max_tokens + bin_width + 1, bin_width)

        plt.figure(figsize=(10, 6))
        plt.hist(token_counts, bins=bins, edgecolor="black")
        plt.xlabel("Tokens per email (BERT tokenizer)")
        plt.ylabel("Number of emails")
        plt.title(f"BERT Token Histogram (bin width = {bin_width})")
        plt.tight_layout()

        hist_path = os.path.join(output_folder, f"bert_token_hist_bin{bin_width}.png")
        plt.savefig(hist_path)
        plt.close()

    return token_counts, stats


if __name__ == "__main__":

    token_stats_with_bert_tokenizer(
        data_folder_path="<path-to-dataset>/Version 4.2.5 c special out",
        output_folder="<path-to-repo>/helper functions/token_counter_using_bert_tokenizer",
        bin_width=100,
        max_tokens=3000,
    )