"""
Purpose:
    Runs the core TF-IDF classification pipeline: for each pair of
    consecutive months, trains on all months so far and tests on the next
    one (a cumulative monthly train/test fold), across Logistic Regression,
    Ridge, SVM and XGBoost. Reports accuracy, per-class F1, confusion
    matrices and vocabulary size for each fold, plus summary plots/reports
    aggregated across all folds.

Thesis reference:
    Implements the monthly cumulative fold methodology (Section 3.2,
    Fig. 3.2). The parameter_sets toggle in main_function() is what
    produced Experiment 1's preprocessing comparison (Section 5.1).

Inputs:
    A folder of monthly .pkl files (data_folder_path), each a dict with
    To/From/CC/Subject/Body/Label columns.

Outputs:
    Per-fold .txt reports and .png plots, plus an all_months_results/
    subfolder of summary plots, written under a per-experiment,
    per-parameter-set results folder.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
import logging
import time
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_score, recall_score, \
    f1_score
import matplotlib.pyplot as plt


def f1_per_class_func(tp_per_class, fp_per_class, fn_per_class):
    """Aggregates TP/FP/FN counts across all folds for each class and computes
    precision, recall and F1 from the totals, returning one F1 score per class."""
    def calculate_precision(tp, fp):
        return tp / (tp + fp) if (tp + fp) > 0 else 0

    def calculate_recall(tp, fn):
        return tp / (tp + fn) if (tp + fn) > 0 else 0

    def calculate_f1(precision, recall):
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    f1_scores = {}
    # Loop through each class (1 to 12)
    for class_key in range(0, 11):
        # Extract TP, FP, FN lists for the current class
        tp_list = tp_per_class.get(class_key, [])
        fp_list = fp_per_class.get(class_key, [])
        fn_list = fn_per_class.get(class_key, [])
        # Calculate aggregate TP, FP, FN for the current class
        aggregate_tp = sum(tp_list)
        aggregate_fp = sum(fp_list)
        aggregate_fn = sum(fn_list)
        # Calculate precision, recall, and F1 score for the current class
        precision = calculate_precision(aggregate_tp, aggregate_fp)
        recall = calculate_recall(aggregate_tp, aggregate_fn)
        f1 = calculate_f1(precision, recall)
        # Store the F1 score for the current class
        f1_scores[class_key] = f1

    return f1_scores


def calculate_distribution(data):
    """Counts how many times each value appears in an iterable, returning a
    dict of value -> count."""
    distribution = {}
    for item in data:
        if item in distribution:
            distribution[item] += 1
        else:
            distribution[item] = 1
    return distribution


def distribution_plotter(distribution_test, distribution_train, training_months, param_results_folder,
                         test_month="not set", loop_count=0):
    """Plots side-by-side bar charts of the train and test label distributions
    for one fold, and saves the figure to param_results_folder."""
    plt.figure(figsize=(15, 5))  # Adjust the figure size as needed

    plt.subplot(1, 2, 1)  # Subplot for y_train
    plt.bar(distribution_train.keys(), distribution_train.values(), color='lightgreen', edgecolor='black')
    plt.title(f'y_train class distribution ({training_months})')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    for x, y in zip(distribution_train.keys(), distribution_train.values()):
        plt.text(x, y + 0.1, str(y), ha='center')
    plt.xticks(list(distribution_train.keys()))

    plt.subplot(1, 2, 2)  # Subplot for y_test
    plt.bar(distribution_test.keys(), distribution_test.values(), color='skyblue', edgecolor='black')
    plt.title(f'y_test class distribution ({test_month})')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    for x, y in zip(distribution_test.keys(), distribution_test.values()):
        plt.text(x, y + 0.1, str(y), ha='center')
    plt.xticks(list(distribution_test.keys()))

    plt.tight_layout()  # Adjust layout to prevent overlap
    # save
    result_file_path = os.path.join(param_results_folder, f'{loop_count + 1}_{test_month}.png')
    plt.savefig(result_file_path)
    plt.close()


def get_month_name(loop_count):
    """Returns month name(s) based on loop count"""
    if loop_count == 1:
        return "jan", "february"
    elif loop_count == 2:
        return "jan_feb", "march"
    elif loop_count == 3:
        return "jan_feb_march", "april"
    elif loop_count == 4:
        return "jan_feb_march_april", "may"
    elif loop_count == 5:
        return "jan_feb_march_april_may", "june"
    elif loop_count == 6:
        return "jan_feb_march_april_may_june", "july"
    elif loop_count == 7:
        return "jan_feb_march_april_may_june_july", "august"
    elif loop_count == 8:
        return "jan_feb_march_april_may_june_july_august", "september"
    elif loop_count == 9:
        return "jan_feb_march_april_may_june_july_august_september", "october"
    elif loop_count == 10:
        return "jan_feb_march_april_may_june_july_august_september_october", "november"
    elif loop_count == 11:
        return "jan_feb_march_april_may_june_july_august_september_october_november", "december"
    else:
        return "unknown_months"


def stemming_func(x_train, x_test, loop_count):
    """Tokenizes and applies Porter stemming to every document in x_train and
    x_test, returning the stemmed versions of both."""
    nltk.download('punkt')
    stemmer = PorterStemmer()

    x_train_stemmed = []
    x_test_stemmed = []

    for document in x_train:
        # Tokenize the document into words
        words = word_tokenize(document)
        # Apply stemming to each word
        stemmed_words = [stemmer.stem(word) for word in words]
        # Join the stemmed words back into a document
        stemmed_document = " ".join(stemmed_words)
        # Append the stemmed document to the list
        x_train_stemmed.append(stemmed_document)

        print(f"loop ({loop_count}). Stemmed train set docs: {len(x_train_stemmed)} out of {len(x_train)}")

    for document in x_test:
        # Tokenize the document into words
        words = word_tokenize(document)
        # Apply stemming to each word
        stemmed_words = [stemmer.stem(word) for word in words]
        # Join the stemmed words back into a document
        stemmed_document = " ".join(stemmed_words)
        # Append the stemmed document to the list
        x_test_stemmed.append(stemmed_document)

        print(f"loop ({loop_count}). Stemmed test set docs: {len(x_test_stemmed)} out of {len(x_test)}")

    return x_train_stemmed, x_test_stemmed


def c_remover(dictionary):
    """This function takes as input a dictionary. it should be either the training or testing month
    Then it finds which index has C in label and removes all the entries with this index."""
    for index in reversed(range(len(dictionary['Label']))):  # reversing order to avoid index shifting
        if dictionary['Label'][index] == "C":
            # Remove the element at that index from all relevant lists
            for key in dictionary:
                dictionary[key].pop(index)

    return dictionary


def multi_pkl_tfidf_fresh_test(data_folder_path, param_results_folder, model, text_representation, stemming,
                               dataset="full", stop_words=None, min_df="Not applicable", ngram_range="Not applicable"):
    """Runs the cumulative monthly train/test loop described in the module
    docstring: loads each month's pkl file in turn, trains on all months seen
    so far, tests on the next month, and writes a report/plot per fold plus
    summary plots aggregated across all folds. Returns the list of training
    vocabulary sizes per fold."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    start_time = time.time()
    loop_durations = []
    training_vocab_sizes = []  # list to store training vocab length of each iteration
    micro_f1_list = []
    per_class_f1_dict = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}
    # Dictionary to store TP, FP, FN for each class
    all_months_tp_dict = {}
    all_months_fp_dict = {}
    all_months_fn_dict = {}
    # List all .pkl files in the folder
    pkl_files = [os.path.join(data_folder_path, f) for f in os.listdir(data_folder_path) if f.endswith('.pkl')]
    pkl_files.sort()
    total_train_dict = {}
    value_to_number = {'A': 0, 'B': 1, 'D': 2, 'F': 3, 'N': 4, 'P': 5, 'R': 6, 'S': 7, 'T': 8, 'U': 9, 'V': 10}
    loop_count = 1
    for i, file_path in enumerate(pkl_files):
        if i + 1 < len(pkl_files):  # Ensures we stop before training on last file as no file to test it.
            training_months, test_month = get_month_name(loop_count)
            # STEP 1: Extend total_train_dict
            new_train_file_path = pkl_files[i]
            print(f"new training file: {new_train_file_path}")
            with open(new_train_file_path, 'rb') as file:
                data_dict = pickle.load(file)

            # removing emails with Class = C due to no interest in classifying them
            data_dict = c_remover(dictionary=data_dict)

            # this part is to add new month in training set
            for key, value_list in data_dict.items():
                if key not in total_train_dict:
                    total_train_dict[key] = value_list  # Create a new list if the key is not present
                else:
                    total_train_dict[key].extend(value_list)  # Extend the list with values from the current data_dict

            # STEP 1.1: Create new column and obtain x_train and y_train
            total_train_dict['To From CC Subject Body'] = [a + ' ' + b + ' ' + c + ' ' + d + ' ' + e for a, b, c, d, e
                                                           in zip(total_train_dict["To"],
                                                                  total_train_dict['From'],
                                                                  total_train_dict['CC'],
                                                                  total_train_dict['Subject'],
                                                                  total_train_dict['Body'])]

            labels = total_train_dict['Label']  # Extract labels
            y_train = [value_to_number[value] for value in labels]  # Converting letters into numbers & storing it as y
            x_train = total_train_dict['To From CC Subject Body']  # Extracting text

            print(f"y_train length: {len(y_train)}")
            print(f"x_train length: {len(x_train)}")

            # STEP 2: Obtain x_test/y_test from {i+1} file
            test_file_path = pkl_files[i + 1]
            print(f"new testing file: {test_file_path}")
            with open(test_file_path, 'rb') as file:
                test_dict = pickle.load(file)

            # removing emails with Class = C due to no interest in classifying them
            test_dict = c_remover(dictionary=test_dict)

            test_dict['To From CC Subject Body'] = [a + ' ' + b + ' ' + c + ' ' + d + ' ' + e for a, b, c, d, e
                                                    in zip(test_dict["To"],
                                                           test_dict['From'],
                                                           test_dict['CC'],
                                                           test_dict['Subject'],
                                                           test_dict['Body'])]

            labels = test_dict['Label']  # Extract labels
            y_test = [value_to_number[value] for value in labels]  # Converting letters into numbers
            x_test = test_dict['To From CC Subject Body']  # Extracting text

            print(f"y_test length: {len(y_test)}")
            print(f"x_test length: {len(x_test)}")

            # STEP 3: Preprocess
            x_train = [text.lower() for text in x_train]
            x_test = [text.lower() for text in x_test]

            if stemming:
                x_train, x_test = stemming_func(x_train, x_test, loop_count)

            # STEP 4: Text representation (Compute feature vectors)
            # TF/IDF
            if text_representation == "tfidf":
                vectorizer = TfidfVectorizer(stop_words=stop_words, min_df=min_df, ngram_range=ngram_range)

                x_train_tfidf_matrix = vectorizer.fit_transform(x_train)  # get tf-idf values
                train_feature_names = vectorizer.get_feature_names_out()  # Get feature names (words)
                training_vocab_sizes.append(len(train_feature_names))
                print(f"x_train_tfidf_matrix.shape: {x_train_tfidf_matrix.shape}")

                x_test_tfidf_matrix = vectorizer.transform(x_test)  # get tf-idf values
                test_feature_names = vectorizer.get_feature_names_out()  # Get feature names (words)
                print(f"x_test_tfidf_matrix.shape: {x_test_tfidf_matrix.shape}")

            # BERT
            elif text_representation == "bert":
                print("placeholder")

            else:
                raise ValueError(f"Unsupported text representation type: {text_representation}")

            # STEP 5: Train classifier
            if model == "logistic_regression":
                classifier = LogisticRegression(multi_class="ovr", solver='liblinear', C=1.5, class_weight="balanced")
                model_params = classifier.get_params()
            elif model == "xgboost":
                classifier = XGBClassifier(objective="multi:softmax", num_class=11)
                model_params = classifier.get_params()
            elif model == "svm":
                classifier = svm.SVC(random_state=42, kernel="poly")
                model_params = classifier.get_params()
            elif model == "ridge_classifier":
                classifier = RidgeClassifier(alpha=1)
                model_params = classifier.get_params()
            else:
                raise ValueError(f"Unsupported model type: {model}")

            classifier.fit(x_train_tfidf_matrix, y_train)  # Train model using TF-IDF matrix and the training labels

            # STEP 6: Make predictions on test set
            y_pred = classifier.predict(x_test_tfidf_matrix)

            # STEP 7: Evaluate the classifier
            accuracy = accuracy_score(y_test, y_pred)  # Calculate the accuracy of the classifier
            print("Validation Accuracy:", accuracy)
            report = classification_report(y_test, y_pred)
            print(report)

            # Compute micro averaged metrics
            micro_precision = precision_score(y_test, y_pred, average='micro')
            micro_recall = recall_score(y_test, y_pred, average='micro')
            micro_f1 = f1_score(y_test, y_pred, average='micro')
            micro_f1_list.append(micro_f1)

            # calculate per class f1 scores and append them to per_class_f1_dict
            for class_id in per_class_f1_dict.keys():
                f1 = f1_score(y_test, y_pred, labels=[class_id], average='macro')
                per_class_f1_dict[class_id].append(f1)

            # confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred)
            # STEP 8: Gather TP FP FN for each class and store them to calculate average f1 of all months later
            tp_dict = {i: 0 for i in range(0, 11)}
            fp_dict = {i: 0 for i in range(0, 11)}
            fn_dict = {i: 0 for i in range(0, 11)}

            for class_idx in range(0, conf_matrix.shape[0]):
                tp = conf_matrix[class_idx, class_idx]
                fp = conf_matrix[:, class_idx].sum() - tp
                fn = conf_matrix[class_idx, :].sum() - tp
                tp_dict[class_idx] += tp
                fp_dict[class_idx] += fp
                fn_dict[class_idx] += fn

            # update outer dictionaries with tp fp fn to keep values for all months
            all_months_tp_dict[test_month] = tp_dict.copy()
            all_months_fp_dict[test_month] = fp_dict.copy()
            all_months_fn_dict[test_month] = fn_dict.copy()
            # calculating target distributions
            distribution_test = calculate_distribution(y_test)
            distribution_train = calculate_distribution(y_train)
            # plotting and saving target distributions
            distribution_plotter(distribution_test,
                                 distribution_train,
                                 test_month=test_month,
                                 loop_count=loop_count,
                                 training_months=training_months,
                                 param_results_folder=param_results_folder)

            # Duration calculation
            dur_seconds = time.time() - start_time
            dur_minutes = round(dur_seconds / 60, 1)
            logger.info(f" Time so far: {dur_seconds} seconds or {dur_minutes} minutes.")
            loop_durations.append(dur_minutes)
            print(f"Training loop durations from start: {loop_durations}")

            # STEP 9: Report
            # class_labels = ["1 (A)", "2 (B)", "3 (D)", "4 (F)", "5 (N)", "6 (P)", "7 (R)", "8 (S)", "9 (T)", "10 (U)",
            #                 "11 (V)"]
            result_file_path = os.path.join(param_results_folder, f'{loop_count + 1}_{test_month}.txt')
            with open(result_file_path, 'w') as result_file:
                # params
                result_file.write(f"Parameters:\n")
                result_file.write(f"training months: {training_months}\n")
                result_file.write(f"testing months: {test_month}\n")
                result_file.write(f"Stemming: {stemming}\n")
                result_file.write(f"Stop words: {stop_words}\n")
                result_file.write(f"min_df: {min_df} (keeps words that appear in at least {min_df} documents)\n")
                result_file.write(f"text representation: {text_representation}\n")
                result_file.write(f"ngram_range: {ngram_range}\n\n")
                result_file.write(f"Classifier: {model}\n")
                result_file.write(f"Classifier params: {model_params}\n")
                result_file.write(f"Training vocab dimensions: {len(train_feature_names)}\n")
                result_file.write(f"x_train_tfidf_matrix.shape:{x_train_tfidf_matrix.shape}\n")
                result_file.write(f"Training loop durations (from start of these params) (mins):{loop_durations}\n")
                # results
                result_file.write(f"\n\nValidation Accuracy: {accuracy}\n")
                result_file.write(f"\n\n{report}")
                result_file.write(f"   micro avg       {micro_precision}      {micro_recall}      {micro_f1}")
                result_file.write("\n\nConfusion Matrix:\n")
                result_file.write(str(conf_matrix))
                result_file.write(f"\n\nTP: {tp_dict}")
                result_file.write(f"\nFP: {fp_dict}")
                result_file.write(f"\nFN: {fn_dict}")
                # distributions
                result_file.write("\n\nDistributions:\n")
                result_file.write("Train Distribution: {}\n".format(distribution_train))
                result_file.write("Test Distribution: {}\n".format(distribution_test))

            # print(f"test file will be {test_file}")
            loop_count += 1

        if dataset == "validation" and loop_count > 3:  # ends function early to run only on validation set
            break

    all_months_folder = os.path.join(param_results_folder, "all_months_results")
    os.makedirs(all_months_folder, exist_ok=True)  # creates folder

    # Training vocab dimensions as training-set increase in monthly increments
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(1, len(training_vocab_sizes) + 1), training_vocab_sizes, color='blue')
    plt.title('Training-set Vocabulary Dimensions')
    plt.xlabel('Training-set size (Months)')
    plt.ylabel('Vocabulary Size')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval, int(yval), va='bottom')
    plt.xticks(range(1, len(training_vocab_sizes) + 1))  # Set x-ticks to integer positions only
    result_file_path = os.path.join(all_months_folder, "all_months_vocab_dimensions.png")  # save
    plt.savefig(result_file_path)
    plt.close()

    # MICRO_F1 AVERAGE
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(training_vocab_sizes) + 1), micro_f1_list)
    plt.title('Micro F1 Average')
    plt.xlabel('Training-set size (Months)')
    plt.ylabel('Micro f1 Average')
    plt.xticks(range(1, len(training_vocab_sizes) + 1))  # Set x-ticks to integer positions only
    result_file_path = os.path.join(all_months_folder, "all_months_micro_f1.png")  # save
    plt.savefig(result_file_path)
    plt.close()

    # TP FP FN per class and per month
    tp_fp_fn_txt_path = os.path.join(all_months_folder, "tp_fp_fn.txt")

    # this is just to also group them per class
    tp_per_class = {key: [] for key in range(0, 11)}
    fp_per_class = {key: [] for key in range(0, 11)}
    fn_per_class = {key: [] for key in range(0, 11)}
    for month in all_months_tp_dict:
        for key in range(0, 11):
            tp_per_class[key].append(all_months_tp_dict[month][key])
    for month in all_months_fp_dict:
        for key in range(0, 11):
            fp_per_class[key].append(all_months_fp_dict[month][key])
    for month in all_months_fn_dict:
        for key in range(0, 11):
            fn_per_class[key].append(all_months_fn_dict[month][key])

    avg_f1_scores = f1_per_class_func(tp_per_class, fp_per_class, fn_per_class)

    with open(tp_fp_fn_txt_path, 'w') as file:
        file.write(f"TP:\n")
        file.write(f"{all_months_tp_dict}\n\n")
        file.write(f"FP:\n")
        file.write(f"{all_months_fp_dict}\n\n")
        file.write(f"FN:\n")
        file.write(f"{all_months_fn_dict}\n\n")
        file.write(f"Grouped per class:\n")
        file.write(f" = = = = = = = = = = = = = = = = = = = = = =\n")
        file.write(f"\nTP:\n")
        for key, values in tp_per_class.items():
            file.write(f"Class {key}: {values}\n")
        file.write(f"\nFP:\n")
        for key, values in fp_per_class.items():
            file.write(f"Class {key}: {values}\n")
        file.write(f"\nFN:\n")
        for key, values in fn_per_class.items():
            file.write(f"Class {key}: {values}\n")
        file.write(f"\nAverage f1 scores:\n")
        file.write(f" = = = = = = = = = = = = = = = = = = = = = =\n")
        for class_key, f1 in avg_f1_scores.items():
            file.write(f"Class {class_key}:{f1}\n")

    # PER CLASS F1 SCORES
    per_class_f1_txt_path = os.path.join(all_months_folder, "f1_scores.txt")
    with open(per_class_f1_txt_path, 'w') as file:
        file.write(f"F1 scores per class as training size increased in monthly increments`: \n\n")
        for class_id, f1_scores in per_class_f1_dict.items():
            file.write(f"Class {class_id}: {f1_scores}\n")

    # Create and save a bar chart for each class
    for class_id, f1_scores in per_class_f1_dict.items():
        plt.figure(figsize=(8, 6))
        bars = plt.bar(range(1, len(f1_scores) + 1), f1_scores, color='skyblue')
        plt.xlabel('Training-set size (Months)')
        plt.ylabel('F1 Score')
        plt.title(f'F1 Scores for Class {class_id}')
        plt.ylim(0, 1)  # Set y-axis range from 0 to 1 for better comparison
        plt.xticks(range(1, len(f1_scores) + 1))  # Set x-axis ticks to match the number of scores
        plt.grid(axis='y')  # Add grid lines for better readability
        for bar in bars:  # Add values at the top of each bar
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.2f}', ha='center', va='bottom')
        file_path = os.path.join(all_months_folder, f'class_{class_id}_f1.png')  # Construct the file path and save
        plt.savefig(file_path)
        plt.close()  # Close the figure to free memory

    return training_vocab_sizes


def main_function(experiment_title,
                  model,
                  text_representation,
                  data_folder_path="<path-to-dataset>/Version 4.2 pkl",
                  desktop_results_folder="<path-to-results>/",
                  dataset="full"):
    """Creates the results folder for this experiment, runs
    multi_pkl_tfidf_fresh_test for each parameter set in parameter_sets, and
    saves comparison plots (vocabulary size per parameter set) plus a total
    duration report."""
    # Experiment 1's preprocessing comparison (Section 5.1, Sets A-D) was produced by
    # toggling these parameter sets one at a time and rerunning - only one set is active
    # (uncommented) at a time.
    parameter_sets = [
        # {
        #     "stemming": False,
        #     "stop_words": None,
        #     "min_df": 1,
        #     "ngram_range": (1, 1),
        #     "params_folder": "param1_nothing"
        # }
        # ,
        # {
        #     "stemming": False,
        #     "stop_words": "english",
        #     "min_df": 1,
        #     "ngram_range": (1, 1),
        #     "params_folder": "param2_stopwords"
        #
        # },
        # {
        #     "stemming": True,
        #     "stop_words": "english",
        #     "min_df": 1,
        #     "ngram_range": (1, 1),
        #     "params_folder": "param3_stopwords_stemming"
        # },
        {
            "stemming": False,
            "stop_words": "english",
            "min_df": 5,
            "ngram_range": (1, 1),
            "params_folder": "param4_stopwords_stemming_mindf"
        }

    ]

    experiment_results_folder = os.path.join(desktop_results_folder, experiment_title)
    os.makedirs(experiment_results_folder, exist_ok=True)  # here we create experiment folder

    # Iterate over each set of parameters and call the function
    start_time_outer = time.time()
    vocab_dimension_dict = {}
    for i, params in enumerate(parameter_sets):
        param_results_folder = os.path.join(experiment_results_folder, params["params_folder"])
        os.makedirs(param_results_folder, exist_ok=True)

        training_vocab_sizes = multi_pkl_tfidf_fresh_test(
            data_folder_path=data_folder_path,
            param_results_folder=param_results_folder,
            stemming=params["stemming"],
            stop_words=params["stop_words"],
            min_df=params["min_df"],
            ngram_range=params["ngram_range"],
            model=model,
            dataset=dataset,
            text_representation=text_representation
        )
        vocab_dimension_dict.update({f"param_set_{i + 1}": training_vocab_sizes})

    # plot vocab_dimension for each param_set
    x_values = range(1, len(training_vocab_sizes) + 1)  # X-axis values (starting from 1)
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'h', '+', 'x']
    plt.figure(figsize=(10, 5))
    # Plot each method using a for loop
    for i, (method, sizes) in enumerate(vocab_dimension_dict.items()):
        plt.plot(x_values, sizes, label=method, marker=markers[i % len(markers)])
    plt.xticks(range(1, len(training_vocab_sizes) + 1))  # Set x-ticks to integer positions only
    plt.title('Vocabulary Size Comparison')
    plt.xlabel('Training-set size (Months)')
    plt.ylabel('Vocabulary Size')
    plt.legend()
    plt.grid(True)
    file_path = os.path.join(experiment_results_folder, 'line_graph.png')  # Construct the file path and save
    plt.savefig(file_path)
    plt.close()

    # Extract the last values for each method for bar chart
    last_values = [sizes[-1] for sizes in vocab_dimension_dict.values()]
    methods = list(vocab_dimension_dict.keys())
    plt.figure(figsize=(10, 5))
    bars = plt.bar(methods, last_values, color='purple')
    plt.title(f'Train-set: {len(training_vocab_sizes)} Months')
    plt.xlabel('Methods')
    plt.ylabel('Vocabulary Size')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval, int(yval), va='bottom')
    file_path = os.path.join(experiment_results_folder, 'bar_chart.png')  # Construct the file path and save
    plt.savefig(file_path)
    plt.close()

    # overall time calculation
    dur_minutes = (time.time() - start_time_outer) / 60
    file_path = os.path.join(experiment_results_folder, 'duration.txt')
    with open(file_path, 'w') as file:
        file.write(f"Whole thing took: {round(dur_minutes, 1)} minutes \n")
        file.write(f"             or : {round(dur_minutes / 60, 1)} hours")
