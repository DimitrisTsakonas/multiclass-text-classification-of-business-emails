"""
Purpose:
    Two standalone corpus-statistics utilities: word_counter (word count per
    email via NLTK tokenizer - max/min/avg) and special_character_counter
    (special-character frequency counts per email).

Thesis reference:
    Exploratory/descriptive corpus statistics, not tied to a specific
    numbered experiment.

Inputs:
    A single month's .pkl file (dict with a 'To From CC Subject Body' key).

Outputs:
    Printed statistics only.
"""
import pickle
from nltk.tokenize import word_tokenize
from tqdm import tqdm


def word_counter(file_path):
    # Load the .pkl file
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    # Extract the documents from the 'To From CC Subject Body' column
    documents = data['To From CC Subject Body']

    # Initialize a list to store word counts of each document
    word_counts = []

    # Iterate over each document and count words
    for document in tqdm(documents, desc="processing documents", unit="docs"):
        tokens = word_tokenize(document)  # Tokenize the document into words
        words = [word for word in tokens if word.isalnum()]  # Filter out non-word tokens (e.g., punctuation)
        word_counts.append(len(words))  # Store the word count for each document

    # Calculate the max, min, and average word counts
    max_word_count = max(word_counts)
    min_word_count = min(word_counts)
    num_of_docs = len(documents)
    avg_word_count = sum(word_counts) / len(word_counts)

    # Print the max, min, and average word counts
    print(f"Max word count: {max_word_count}")
    print(f"Min word count: {min_word_count}")
    print(f"Average word count: {avg_word_count:.2f}")

    print(f"number of docs: {num_of_docs}")

    return word_counts


def special_character_counter(file_path):
    # Load the .pkl file
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    # Extract the documents from the 'To From CC Subject Body' column
    documents = data['To From CC Subject Body']

    # Define the special characters to count
    special_chars = ['?', '.', '~', '!', ',', ':', ';', '-', '@', '#', '$', '%', '^', '&', '*', '(', ')', '[', ']', '{',
                     '}', '+', '=', '<', '>', '"', "'"]

    # Initialize a dictionary to store the counts for each special character
    special_char_counts = {char: 0 for char in special_chars}

    # Iterate over each document and count special characters
    for document in tqdm(documents, desc="processing documents", unit="docs"):
        # Iterate through each character in the document
        for char in document:
            if char in special_char_counts:
                special_char_counts[char] += 1  # Increment the count for that special character

    # Calculate the total number of special characters
    total_special_chars = sum(special_char_counts.values())

    # Print the counts of special characters
    print(f"Total special characters: {total_special_chars}")
    for char, count in special_char_counts.items():
        print(f"{char}: {count}")

    return special_char_counts


special_char_counts = special_character_counter(
    file_path=r"<path-to-dataset>\Version 4.2.1 pkl stemmed cout\01_January_23_version_4.2.1.pkl")