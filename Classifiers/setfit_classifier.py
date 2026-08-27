"""
Purpose:
    Exploratory SetFit (few-shot sentence-transformer) classification
    experiment on a single train/test month pair, iterating over different
    class-balancing/sample-size strategies (limit_classes, later replaced by
    balanced_limited_sample) as an alternative to the TF-IDF approach used
    elsewhere in this repo.

Thesis reference:
    Not part of the 10 reported experiments - SetFit is never mentioned in
    the thesis text. Kept as a record of exploratory work into few-shot
    embedding-based classification; real output from 8 iterative runs exists
    in results/79_setfit/.

Inputs:
    Two single-month .pkl files (train + test), unlike the cumulative
    monthly-fold setup used by the other classifiers.

Outputs:
    A single evaluation report .txt file per run under results/79_setfit/.
"""
import pickle
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import torch
import os
import time
from collections import defaultdict, Counter
import random
import math
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)
print("CUDA Available:", torch.cuda.is_available())
print("Device being used:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))


start_time = time.time()


def limit_classes(x, y, class_limits):
    """
    Earlier sampling approach, later superseded by balanced_limited_sample
    below (see the commented-out call sites) - kept here for reference.

    Limits number of examples per class based on class_limits dictionary.

    class_limits = {
        1: 5000,   # Limit class 1 to 5000 examples
        5: 5000,   # Limit class 5 to 5000 examples
        # other classes not listed will be kept entirely
    }
    """
    class_data = {}  # class -> list of (text, label)
    for text, label in zip(x, y):
        class_data.setdefault(label, []).append(text)  # setdedault creates key if it doesnt exist.

    x_out, y_out = [], []
    for label, texts in class_data.items():
        limit = class_limits.get(label, len(texts))  # if label is found in class_limits, returns value, else len(texts)
        sampled = random.sample(texts, min(len(texts), limit))
        x_out.extend(sampled)
        y_out.extend([label] * len(sampled))

    return x_out, y_out


def balanced_limited_sample(x, y, total_limit=500, min_per_class=2, max_per_class=100, seed=42):
    random.seed(seed)
    label_counts = Counter(y)
    class_indices = defaultdict(list)
    for idx, label in enumerate(y):
        class_indices[label].append(idx)

    total_available = sum(label_counts.values())
    raw_allocations = {
        label: max(min_per_class, min(max_per_class, round((count / total_available) * total_limit)))
        for label, count in label_counts.items()
    }

    # Enforce total sum to equal `total_limit`
    total_raw = sum(raw_allocations.values())
    if total_raw > total_limit:
        # Normalize down
        scaling_factor = total_limit / total_raw
        adjusted_allocations = {
            label: max(min_per_class, min(max_per_class, math.floor(alloc * scaling_factor)))
            for label, alloc in raw_allocations.items()
        }

        # Fix rounding errors
        diff = total_limit - sum(adjusted_allocations.values())
        if diff > 0:
            # Add leftover to biggest remaining allowed classes
            leftover_labels = sorted(
                adjusted_allocations.items(),
                key=lambda kv: label_counts[kv[0]],  # prioritize bigger classes
                reverse=True
            )
            for label, _ in leftover_labels:
                available = min(max_per_class, len(class_indices[label])) - adjusted_allocations[label]
                if available > 0:
                    adjusted_allocations[label] += 1
                    diff -= 1
                    if diff == 0:
                        break
    else:
        adjusted_allocations = raw_allocations

    # Final sampling
    sampled_indices = []
    for label, count in adjusted_allocations.items():
        indices = class_indices[label]
        if len(indices) >= count:
            sampled_indices.extend(random.sample(indices, count))
        else:
            sampled_indices.extend(indices)

    random.shuffle(sampled_indices)
    x_sampled = [x[i] for i in sampled_indices]
    y_sampled = [y[i] for i in sampled_indices]
    return x_sampled, y_sampled


def load_month_data(month_file):
    with open(month_file, 'rb') as file:
        data = pickle.load(file)
    return data


train_month = load_month_data(r"<path-to-dataset>\Version 4.2.5 c special out\01_January_23_version_4.2.5.pkl")
test_month = load_month_data(r"<path-to-dataset>\Version 4.2.5 c special out\02_February_23_version_4.2.5.pkl")

x_train = train_month["To From CC Subject Body"]
y_train = train_month["Label"]
# class_limits = {1: 7000, 5: 7000}
all_labels = set(y_train)
class_limits = {label: 50 for label in all_labels}  # i think its not needed now

# x_train_balanced, y_train_balanced = limit_classes(x_train, y_train, class_limits)
x_train_balanced, y_train_balanced = balanced_limited_sample(x_train, y_train, total_limit=500, min_per_class=10,
                                                             max_per_class=200)
print("🔍 Class distribution in balanced train+val set:")
print("Total samples:", len(x_train_balanced))
class_counts = Counter(y_train_balanced)
for cls, count in sorted(class_counts.items()):
    print(f"  Class {cls}: {count} examples")

# ✅ Split Month 1 into Train (90%) & Validation (10%) with stratification
train_texts, val_texts, train_labels, val_labels = train_test_split(
    x_train_balanced, y_train_balanced,
    test_size=0.1,
    stratify=y_train_balanced,  # ✅ Keeps class distribution the same
    random_state=42
)


print("🔍 Class distribution in balanced train set:")
train_class_counts = Counter(train_labels)
val_class_counts = Counter(val_labels)
for cls, count in sorted(train_class_counts.items()):
    print(f"  Class {cls}: {count} examples")
print("🔍 Class distribution in balanced validation set:")
for cls, count in sorted(val_class_counts.items()):
    print(f"  Class {cls}: {count} examples")


# Convert to Hugging Face Dataset format *** CHANGE BODY TO ADD TOFROMSBJ ETC ****
df_train = Dataset.from_dict({"text": train_texts, "label": train_labels})
df_val = Dataset.from_dict({"text": val_texts, "label": val_labels})

# Reduce test set class counts
x_test = test_month["To From CC Subject Body"]
y_test = test_month["Label"]

all_test_labels = set(y_test)
test_class_limits = {label: 2000 for label in all_test_labels}  ## i think i dont need now

# x_test_balanced, y_test_balanced = limit_classes(x_test, y_test, test_class_limits)
x_test_balanced, y_test_balanced = balanced_limited_sample(x_test, y_test, total_limit=10000, min_per_class=1,
                                                           max_per_class=9999999)


print("🔍 Class distribution in balanced test set:")
print("Total samples:", len(x_test_balanced))
test_class_counts = Counter(y_test_balanced)
for cls, count in sorted(test_class_counts.items()):
    print(f"  Class {cls}: {count} examples")
# Convert to Hugging Face Dataset
df_test = Dataset.from_dict({"text": x_test_balanced, "label": y_test_balanced})
# df_test = Dataset.from_dict({"text": test_month["To From CC Subject Body"], "label": test_month["Label"]})


model_name = "sentence-transformers/paraphrase-mpnet-base-v2"
model = SetFitModel.from_pretrained(model_name)

args = TrainingArguments(
            batch_size=2,
            num_epochs=2,
            sampling_strategy="undersampling",
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )


def compute_metrics(eval_preds, model_name=None, train_class_counts=None, val_class_counts=None):
    predictions, labels = eval_preds

    # Basic metrics
    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average='macro')
    micro_f1 = f1_score(labels, predictions, average='micro')
    weighted_f1 = f1_score(labels, predictions, average='weighted')
    macro_precision = precision_score(labels, predictions, average='macro', zero_division=0)
    macro_recall = recall_score(labels, predictions, average='macro', zero_division=0)

    # Classification report and confusion matrix
    class_report = classification_report(labels, predictions, digits=2, zero_division=0)
    cm = confusion_matrix(labels, predictions)

    # Compute TP, FP, FN
    tp = {}
    fp = {}
    fn = {}

    num_classes = cm.shape[0]
    for i in range(num_classes):
        tp[i] = cm[i][i]
        fp[i] = sum(cm[:, i]) - cm[i][i]
        fn[i] = sum(cm[i]) - cm[i][i]

    # Output path
    output_dir = r"<path-to-results>\79_setfit"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "79_8_setfit_stratified_train500min10max200_test10000.txt")
    dur_seconds = time.time() - start_time
    dur_minutes = round(dur_seconds / 60, 1)
    with open(output_path, "w") as f:
        if model_name:
            f.write(f"Model: {model_name}\n\n")
            f.write(f"Duration (mins): {dur_minutes}\n\n")
        if train_class_counts:
            f.write("Training Set Class Distribution:\n")
            for cls, count in sorted(train_class_counts.items()):
                f.write(f"  Class {cls}: {count} examples\n")
            f.write("\n")

        if val_class_counts:
            f.write("Validation Set Class Distribution:\n")
            for cls, count in sorted(val_class_counts.items()):
                f.write(f"  Class {cls}: {count} examples\n")
            f.write("\n")

        f.write("Classification Report (Per Class):\n")
        f.write(class_report + "\n\n")

        f.write("Confusion Matrix:\n")
        for row in cm:
            f.write(" ".join(map(str, row)) + "\n")
        f.write("\n")

        f.write(f"True Positives (TP): {tp}\n")
        f.write(f"False Positives (FP): {fp}\n")
        f.write(f"False Negatives (FN): {fn}\n")

    print(f"\n Saved metrics report to: {output_path}")

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall
    }


def metric_fn(*args):
    # Called during training (1 arg: tuple of (preds, labels))
    if len(args) == 1:
        return compute_metrics(args[0], model_name, train_class_counts, val_class_counts)

    # Called during evaluation (2 args: preds, labels)
    elif len(args) == 2:
        preds, labels = args
        return compute_metrics((preds, labels), model_name, train_class_counts, val_class_counts)


trainer = Trainer(
            model=model,
            args=args,
            train_dataset=df_train,
            eval_dataset=df_val,
            metric=metric_fn,
            column_mapping={"text": "text", "label": "label"},
        )

print("starting training")
trainer.train()
print("finished training, proceeding with evaluation")
metrics = trainer.evaluate(df_test)
print(metrics)

dur_seconds = time.time() - start_time
dur_minutes = round(dur_seconds / 60, 1)
print(f"total duration in minutes: {dur_minutes}")