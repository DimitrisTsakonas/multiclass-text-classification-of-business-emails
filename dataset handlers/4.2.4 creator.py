"""
Purpose:
    Converts Version 4.2 monthly pkl files into Version 4.2.4: removes class
    C, keeps only the combined text column and the label, encodes labels as
    numbers, removes special characters, lowercases and stems the text.
    Prints before/after counts of special characters and uppercase letters
    as a manual sanity check while running.

Thesis reference:
    Dataset preparation for the special-character-removal experiment
    (Section 5.5), combined with the stemming/lowercasing used elsewhere.

Inputs:
    A folder of Version 4.2 monthly .pkl files (data_folder_path).

Outputs:
    One Version 4.2.4 .pkl file per month, saved alongside the input files
    under a sibling "Version 4.2.4 stemmed c special out" folder.
"""

import pickle
import os
import logging
import time
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from tqdm import tqdm


def stemming_func(documents):
    stemmer = PorterStemmer()

    documents_stemmed = []

    for document in tqdm(documents, desc="Stemming documents", unit="doc"):
        # Tokenize the document into words
        words = word_tokenize(document)
        # Apply stemming to each word
        stemmed_words = [stemmer.stem(word) for word in words]
        # Join the stemmed words back into a document
        stemmed_document = " ".join(stemmed_words)
        # Append the stemmed document to the list
        documents_stemmed.append(stemmed_document)

    return documents_stemmed


def c_remover(dictionary):
    """This function takes as input a dictionary. it should be either the training or testing month
    Then it finds which index has C in label and removes all the entries with this index."""
    for index in reversed(range(len(dictionary['Label']))):  # reversing order to avoid index shifting
        if dictionary['Label'][index] == "C":
            # Remove the element at that index from all relevant lists
            for key in dictionary:
                dictionary[key].pop(index)

    return dictionary


def remove_special_characters_manually(text):
    # Define a list of special characters to remove
    special_chars = ['?', '.', '~', '!', ',', ':', ';', '-', '@', '#', '$', '%', '^', '&', '*', '(', ')', '[', ']', '{',
                     '}', '+', '=', '<', '>', '"', "'"]

    # Loop through each character in the text and replace it if it is in the special_chars list
    cleaned_text = ''.join([char for char in text if char not in special_chars])

    return cleaned_text


def special_chars_counter(list_of_docs):
    special_chars = ['?', '.', '~', '!', ',', ':', ';', '-', '@', '#', '$', '%', '^', '&', '*', '(', ')', '[', ']', '{',
                     '}', '+', '=', '<', '>', '"', "'"]
    # Initialize a dictionary to store the counts for each special character
    special_char_counts = {char: 0 for char in special_chars}

    # Iterate over each document and count special characters
    for document in list_of_docs:
        # Iterate through each character in the document
        for char in document:
            if char in special_char_counts:
                special_char_counts[char] += 1  # Increment the count for that special character

    return special_char_counts


def check_uppercase_in_documents(documents):
    """Check each document in the list and see if it contains any uppercase letters."""
    results = []
    for doc in documents:
        # Check if any character in the document is uppercase
        results.append(any(char.isupper() for char in doc))

    # Count how many True values are in the results list
    true_count = results.count(True)

    return true_count


def creator_of_4_2_4(data_folder_path):
    """takes 4.2 version of data as input and creates a new version 4.2.4.
    1) Removes class C.
    2) Keeps only ['To From CC Subject Body'] and ['labels'].
    3) Encodes ['labels'] into numbers
    4) Removes special characters
    5) Lower
    6) Stemming

    saves a .pkl file for each month in similar fashion with 4.2 but now its called 4.2.4

    """
    start_time = time.time()
    logger = logging.getLogger(__name__)
    nltk.download('punkt')

    pkl_files = [os.path.join(data_folder_path, f) for f in os.listdir(data_folder_path) if f.endswith('.pkl')]
    pkl_files.sort()
    for i, file_path in enumerate(tqdm(pkl_files, desc="Processing files")):
        with open(pkl_files[i], 'rb') as file:
            data_dict = pickle.load(file)

        # 1) c removal
        data_dict = c_remover(dictionary=data_dict)  # removing emails Class = C due to no interest in classifying them
        # 2) ['To From CC Subject Body'] and ['labels']
        data_dict['To From CC Subject Body'] = [a + ' ' + b + ' ' + c + ' ' + d + ' ' + e for a, b, c, d, e
                                                in zip(data_dict["To"],
                                                       data_dict['From'],
                                                       data_dict['CC'],
                                                       data_dict['Subject'],
                                                       data_dict['Body'])]
        # 3) letter -> num
        labels = data_dict['Label']  # Extract labels
        value_to_number = {'A': 0, 'B': 1, 'D': 2, 'F': 3, 'N': 4, 'P': 5, 'R': 6, 'S': 7, 'T': 8, 'U': 9, 'V': 10}
        data_dict['Label'] = [value_to_number[value] for value in labels]  # letters into numbers & update dict in place

        # 4) special chars removed
        # print how many special chars exist in my docs
        number_of_specialchars = special_chars_counter(data_dict['To From CC Subject Body'])
        print(f"my documents currently have {number_of_specialchars} special characters")
        print(f"starting special chars removal process...")
        # remove special chars
        data_dict['To From CC Subject Body'] = [remove_special_characters_manually(text) for text in
                                                data_dict['To From CC Subject Body']]
        # print how many texts have special chars (should be zero)
        number_of_specialchars_2 = special_chars_counter(data_dict['To From CC Subject Body'])
        print(f"my documents now have {number_of_specialchars_2} special characters after removal process.")


        # 5) .lower and save it in place
        emails_with_upper = check_uppercase_in_documents(data_dict['To From CC Subject Body'])
        print(f"There are {emails_with_upper} emails that contain upper case letters")
        print("Starting .lower process...")
        data_dict['To From CC Subject Body'] = [text.lower() for text in data_dict['To From CC Subject Body']]
        # check how many upper case chars exist
        emails_with_upper_after = check_uppercase_in_documents(data_dict['To From CC Subject Body'])
        print(f"There are {emails_with_upper_after} emails that contain upper case letters AFTER preprocessing")
        # 6) stemming and save it in place
        data_dict['To From CC Subject Body'] = stemming_func(data_dict['To From CC Subject Body'])

        # save to new file
        new_dict = {key: data_dict[key] for key in ['To From CC Subject Body', 'Label']}
        new_file_path = file_path.replace("Version 4.2 pkl", "Version 4.2.4 stemmed c special out").replace(
            "version_4.2.pkl", "version_4.2.4.pkl")
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(new_file_path), exist_ok=True)

        with open(new_file_path, 'wb') as file:
            pickle.dump(new_dict, file)

    dur_seconds = time.time() - start_time
    dur_minutes = round(dur_seconds / 60, 1)
    logger.info(f" Time so far: {dur_seconds} seconds or {dur_minutes} minutes.")

creator_of_4_2_4(data_folder_path="<path-to-dataset>/Version 4.2 pkl")
