"""
Purpose:
    Converts Version 4.2 monthly pkl files into Version 4.2.1: removes class
    C, keeps only the combined text column and the label, encodes labels as
    numbers, lowercases and stems the text.

Thesis reference:
    Dataset preparation (Section 4.2.1) for the stemmed/lowercased TF-IDF
    experiments.

Inputs:
    A folder of Version 4.2 monthly .pkl files (data_folder_path).

Outputs:
    One Version 4.2.1 .pkl file per month, saved alongside the input files
    under a sibling "Version 4.2.1 pkl stemmed cout" folder.
"""

import pickle
import os
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from tqdm import tqdm


def c_remover(dictionary):
    """This function takes as input a dictionary. it should be either the training or testing month
    Then it finds which index has C in label and removes all the entries with this index."""
    for index in reversed(range(len(dictionary['Label']))):  # reversing order to avoid index shifting
        if dictionary['Label'][index] == "C":
            # Remove the element at that index from all relevant lists
            for key in dictionary:
                dictionary[key].pop(index)

    return dictionary


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


def stemmer_lower_numbers_func(data_folder_path="<path-to-dataset>/Version 4.2 pkl"):
    """takes 4.2 version of data as input and creates a new version 4.2.1
    1) Removes class C.
    2) Keeps only ['To From CC Subject Body'] and ['labels'].
    3) Encodes ['labels'] into numbers
    4) .lower
    5) stemming

    saves a .pkl file for each month in similar fashion with 4.2 but now its called 4.2.1

    """
    nltk.download('punkt')

    pkl_files = [os.path.join(data_folder_path, f) for f in os.listdir(data_folder_path) if f.endswith('.pkl')]
    pkl_files.sort()
    monthly_size_no_c = []
    for i, file_path in enumerate(pkl_files):
        with open(pkl_files[i], 'rb') as file:
            data_dict = pickle.load(file)

        # 1) c removal
        data_dict = c_remover(dictionary=data_dict)  # removing emails Class = C due to no interest in classifying them
        monthly_size_no_c.append(len(data_dict["Label"]))  # saving monthly size to check how many emails we have in end
        # 2)
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

        # 4) .lower and save it in place
        data_dict['To From CC Subject Body'] = [text.lower() for text in data_dict['To From CC Subject Body']]
        # 5) stemming and save it in place
        data_dict['To From CC Subject Body'] = stemming_func(data_dict['To From CC Subject Body'])

        # save to new file
        new_dict = {key: data_dict[key] for key in ['To From CC Subject Body', 'Label']}
        new_file_path = file_path.replace("Version 4.2 pkl", "Version 4.2.1 pkl stemmed cout").replace(
            "version_4.2.pkl", "version_4.2.1.pkl")
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(new_file_path), exist_ok=True)

        with open(new_file_path, 'wb') as file:
            pickle.dump(new_dict, file)

    print(monthly_size_no_c)


stemmer_lower_numbers_func()
