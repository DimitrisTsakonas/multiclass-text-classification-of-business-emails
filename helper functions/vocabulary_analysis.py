"""
Purpose:
    Grab-bag of standalone vocabulary/text-composition exploration
    utilities: special-character frequency counting across a dataset
    version, per-month mean-TF-IDF feature ranking, and per-class word-cloud
    visualization.

Thesis reference:
    Exploratory/descriptive dataset-composition analysis, not tied to a
    specific numbered experiment.

Inputs:
    Monthly .pkl files (dict with 'Label' and 'To From CC Subject Body'
    keys), depending on which function is called.

Outputs:
    Printed statistics (and, for monthly_worldcloud_per_class, an on-screen
    matplotlib word-cloud figure); no files are written to disk. Only
    monthly_tfidf_vocab_analyser runs automatically on import - the other
    functions are utilities meant to be called as needed.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
import logging
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
pd.set_option('display.max_rows', 1000)


def special_chars_counter(list_of_docs):
    print("starting special chars counter...")
    special_chars = ['?', '.', '~', '!', ',', ':', ';', '-', '@', '#', '$', '%', '^', '&', '*', '(', ')', '[', ']', '{',
                     '}', '+', '=', '<', '>', '"', "'"]

    # Initialize a dictionary to store the counts for each special character
    special_char_counts = {char: 0 for char in special_chars}

    # Iterate over each document and count special characters
    for document in list_of_docs:
        for char in document:
            if char in special_char_counts:
                special_char_counts[char] += 1  # Increment the count for that special character

    # Print the characters that were found and their counts
    print("\nSpecial Characters Found:")
    for char, count in special_char_counts.items():
        if count > 0:  # Only print characters that were found
            print(f"Character: '{char}' → Found {count} times")

    print(f"total special chars found: {special_char_counts}")


def tfidf_vocab_analyser(data_folder_path):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    start_time = time.time()
    total_emails_dict = {}
    pkl_files = [os.path.join(data_folder_path, f) for f in os.listdir(data_folder_path) if f.endswith('.pkl')]
    pkl_files.sort()

    for i, file_path in enumerate(pkl_files):
        new_train_file_path = pkl_files[i]
        print(f"new training file: {new_train_file_path}")

        with open(new_train_file_path, 'rb') as file:
            data_dict = pickle.load(file)

        # this part is to add new month in training set
        for key, value_list in data_dict.items():
            if key not in total_emails_dict:
                total_emails_dict[key] = value_list  # Create a new list if the key is not present
            else:
                total_emails_dict[key].extend(value_list)  # Extend the list with values from the current data_dict

        print(f"finished file i={i}")

    x_train = total_emails_dict['To From CC Subject Body']
    print(f"x_train length = {len(x_train)}")

    special_chars_counter(x_train)

    dur_seconds = time.time() - start_time
    dur_minutes = round(dur_seconds / 60, 1)
    logger.info(f" Time so far: {dur_seconds} seconds or {dur_minutes} minutes.")


# tfidf_vocab_analyser(data_folder_path="M:/Diplomatiki/dataset/Version 4.2.3 c special out")


def monthly_tfidf_vocab_analyser(pkl_file_path):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    start_time = time.time()

    with open(pkl_file_path, 'rb') as file:
        total_train_dict = pickle.load(file)

    y_train = total_train_dict['Label']
    x_train = total_train_dict['To From CC Subject Body']

    print("starting tfidf calculation...")
    vectorizer = TfidfVectorizer(stop_words="english", min_df=5, ngram_range=(1, 1))
    x_train_tfidf_matrix = vectorizer.fit_transform(x_train)  # get tf-idf values
    train_feature_names = vectorizer.get_feature_names_out()
    print(f"x_train_tfidf_matrix.shape: {x_train_tfidf_matrix.shape}")
    # Compute the mean TF-IDF score for each feature
    tfidf_mean_scores = np.asarray(x_train_tfidf_matrix.mean(axis=0)).flatten()

    tfidf_df = pd.DataFrame({'Feature': train_feature_names, 'Mean TF-IDF Score': tfidf_mean_scores})
    tfidf_df = tfidf_df.sort_values(by="Mean TF-IDF Score", ascending=False)

    print(tfidf_df.tail(300))

    end_time = time.time()
    dur_minutes = (end_time-start_time)/60
    print(f"duration (mins):{round(dur_minutes, 2)}")


monthly_tfidf_vocab_analyser(
    pkl_file_path=r"<path-to-dataset>/Version 4.2.4 stemmed c special out/01_January_23_version_4.2.4.pkl")


def monthly_worldcloud_per_class(pkl_file_path, class_number):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    start_time = time.time()

    with open(pkl_file_path, 'rb') as file:
        total_train_dict = pickle.load(file)

    df = pd.DataFrame.from_dict(total_train_dict)
    df_filtered = df[df["Label"] == class_number]
    print(df_filtered.head(5))
    docs_of_filtered_class = df_filtered['To From CC Subject Body'].tolist()

    count_vectorizer = CountVectorizer(stop_words="english", min_df=5, ngram_range=(1, 1))
    X_count = count_vectorizer.fit_transform(docs_of_filtered_class)
    feature_names = count_vectorizer.get_feature_names_out()
    word_frequencies = X_count.toarray().sum(axis=0)
    wordcount_freq = pd.DataFrame(list(zip(feature_names, word_frequencies)), columns=["Feature", "Class Raw Frequency"])
    wordcount_freq = wordcount_freq.sort_values(by="Class Raw Frequency", ascending=False)
    print(f"number of features: {wordcount_freq.shape[0]}")
    print("top 100:")
    print(wordcount_freq.head(100))
    print("bottom 100:")
    print(wordcount_freq.tail(100))

    word_freq_dict = dict(zip(wordcount_freq['Feature'], wordcount_freq['Class Raw Frequency']))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq_dict)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Turn off axis
    plt.show()

    end_time = time.time()
    dur_minutes = (end_time-start_time)/60
    print(f"duration (mins):{round(dur_minutes, 2)}")


# monthly_worldcloud_per_class(
#     pkl_file_path=r"M:/Diplomatiki/dataset/Version 4.2.5 c special out/02_February_23_version_4.2.5.pkl",
#     class_number=1)
