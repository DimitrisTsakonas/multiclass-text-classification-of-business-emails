"""
Purpose:
    Converts Version 4.2 monthly pkl files into Version 4.2.3: removes class
    C, keeps only the combined text column and the label, and encodes labels
    as numbers.

Thesis reference:
    Intended as the special-character-removed input for Experiment 5/6
    (Section 5.5/5.6). Despite the "special out" folder name, this function
    never actually removes special characters - that step was never added
    to the code, which is why the special-character counts reported at the
    time showed no change. Version 4.2.5 is the corrected redo; this
    version is kept for the historical record.

Inputs:
    A folder of Version 4.2 monthly .pkl files (data_folder_path).

Outputs:
    One Version 4.2.3 .pkl file per month, saved alongside the input files
    under a sibling "Version 4.2.3 c special out" folder.
"""

import pickle
import os
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


def creator_of_4_2_3(data_folder_path="<path-to-dataset>/Version 4.2 pkl"):
    """takes 4.2 version of data as input and creates a new version 4.2.3.
    1) Removes class C.
    2) Keeps only ['To From CC Subject Body'] and ['labels'].
    3) Encodes ['labels'] into numbers

    saves a .pkl file for each month in similar fashion with 4.2 but now its called 4.2.3

    """
    pkl_files = [os.path.join(data_folder_path, f) for f in os.listdir(data_folder_path) if f.endswith('.pkl')]
    pkl_files.sort()
    for i, file_path in enumerate(tqdm(pkl_files, desc="Processing files")):
        with open(pkl_files[i], 'rb') as file:
            data_dict = pickle.load(file)

        # 1) c removal
        data_dict = c_remover(dictionary=data_dict)  # removing emails Class = C due to no interest in classifying them
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

        # save to new file
        new_dict = {key: data_dict[key] for key in ['To From CC Subject Body', 'Label']}
        new_file_path = file_path.replace("Version 4.2 pkl", "Version 4.2.3 c special out").replace(
            "version_4.2.pkl", "version_4.2.3.pkl")
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(new_file_path), exist_ok=True)

        with open(new_file_path, 'wb') as file:
            pickle.dump(new_dict, file)


creator_of_4_2_3()
