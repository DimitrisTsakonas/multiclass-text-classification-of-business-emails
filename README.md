# Multiclass Text Classification of Business Emails

Code accompanying the MSc thesis of Dimitrios Tsakonas, MSc Data Science, University of the Peloponnese & NCSR "Demokritos", April 2026. The task: classify incoming/outgoing emails at a real shipbroking company into one of 11 business categories, most of them severely underrepresented.

## About

The thesis evaluates TF-IDF and embedding-based (BERT, Nomic Embed Text) representations, several classifiers (Logistic Regression, Ridge, SVM, XGBoost), and a two-stage meta-classifier for the two most commonly confused categories, across 10 experiments run on a real, privately-held email dataset. The dataset itself is not included in this repository - see "Running the code" below.

## Repository structure

```
email_parser/
├── dataset_readme.txt        Dataset version history/notes
│
├── dataset handlers/         parsers.py, csv_creator.py, csv_handler.py (raw email
│                              tagging/extraction/cleaning), plus 10 scripts, one per
│                              dataset version (4.2.1 → 4.2.5.4): preprocessing, label
│                              encoding, stemming, embeddings, etc.
│
├── Classifiers/               main.py (the original TF-IDF classification pipeline,
│                              monthly cumulative folds) plus 8 classifier scripts
│                              covering all 10 experiments, plus two unreported
│                              exploratory scripts (clearly labeled as such)
│
└── helper functions/          Standalone analysis/statistics/plotting utilities:
    ├── report figures/        Scripts that generate the thesis's Chapter 4/5 figures
    ├── token_counter_using_bert_tokenizer/
    ├── token_counter_using_tfidf_tokenizer/
    └── *.py                   Vocabulary analysis, word counts, significance tests, etc.
```

Every kept file has a header docstring (Purpose / Thesis reference / Inputs / Outputs) pointing to the exact section, figure, or table it relates to.

## Label taxonomy

11 categories remain after class C (Circular) is dropped during preprocessing:

| Label | Category               | Numeric code |
|-------|-------------------------|:------------:|
| A     | Accounting               | 0 |
| B     | Business                 | 1 |
| D     | Disbursement Accounts     | 2 |
| F     | Fixtures                 | 3 |
| N     | Negotiations              | 4 |
| P     | Positions                 | 5 |
| R     | Reports                  | 6 |
| S     | Sales                    | 7 |
| T     | Tankers                  | 8 |
| U     | Unknown                  | 9 |
| V     | Various                  | 10 |

## Experiment map

| # | Experiment | Thesis section | Implemented in |
|---|---|---|---|
| 1 | Initial Exploration | 5.1 | `Classifiers/main.py`, `helper functions/report figures/exp1_*.py` |
| 2 | Method Evaluation | 5.2 | `helper functions/report figures/exp2_conf_matrix.py` |
| 3 | Full Scale Evaluation | 5.3 | `helper functions/report figures/confusion_matrix_creator.py`, `exp3_conf_matrix.py`, `exp3_micro_macro_graph.py` |
| 4 | Embeddings | 5.4 | `Classifiers/bert_classifier.py` |
| 5 | Special Characters | 5.5 | `dataset handlers/4.2.3 creator.py`, `4.2.5 creator.py` |
| 6 | Embeddings Variations | 5.6 | `Classifiers/bert_classifier.py`, `helper functions/non parametric test.py` |
| 7 | Nomic Embeddings | 5.7 | `Classifiers/bert_classifier.py`, `helper functions/token_counter_using_bert_tokenizer/bert_token_counter.py` |
| 8 | Combined Representation | 5.8 | `Classifiers/combo_classifier.py`, `helper functions/missclassifications_handler.py`, `helper functions/report figures/venn_creator.py` |
| 9 | Feature Reduction | 5.9 | `Classifiers/tfidf_classifier.py` |
| 10 | Meta Classifier | 5.10 | `Classifiers/tfidf_classifier_first_stage_3_4combined.py`, `second_stage_binary_tfidf_classifier.py`, `tfidf_classifier_with_meta.py` |

Two `Classifiers/` scripts are exploratory work that never made it into the thesis text (`tfidf_classifier_2vsall_binary.py`, `setfit_classifier.py`).

## Running the code

The dataset (real company emails) is not included in this repository and can't be shared. Every script that reads dataset files uses placeholder paths like `<path-to-dataset>/...` and `<path-to-results>/...` in place of the author's local paths - this repository is offered as a reference implementation of the thesis's methodology, not a plug-and-run tool.
