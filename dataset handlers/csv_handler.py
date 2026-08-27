"""
Purpose:
    Two dataset-cleaning steps used in sequence: nul_removal() strips stray
    NUL bytes from a CSV, and csv_to_dict() + dict_to_pkl() load a CSV into
    a dict and pickle it for use by the classifiers.

Thesis reference:
    Section 4.2.1 - Dataset. nul_removal() produces the null-stripped CSV
    (Version 4.1); csv_to_dict() + dict_to_pkl() produce the final pickled
    dataset (Version 4.2) used in the reported experiments.

Inputs:
    nul_removal(): a raw CSV (input_csv_file).
    csv_to_dict(): a null-stripped CSV (csv_file).
    dict_to_pkl(): a dict (dictionary) to save.

Outputs:
    nul_removal(): a cleaned CSV (output_file).
    csv_to_dict(): a dict of column name -> list of values.
    dict_to_pkl(): a .pkl file (output_pkl_path).

Notes:
    nul_removal() was run once to produce the Version 4.1 no-null csv, then
    its example call below was commented out - it is not invoked by this
    file's active code, only csv_to_dict()/dict_to_pkl() run automatically.
"""

import csv
import logging
import time
import pickle


def nul_removal(input_csv_file, output_file):
    """Removes stray NUL byte characters from a CSV file (some rows were encoded
    incorrectly) and writes the cleaned rows to a new CSV file."""
    csv.field_size_limit(2147483647)
    with open(input_csv_file, 'r', encoding='utf-8') as csv_in, open(output_file, 'w', newline='', encoding='utf-8') as csv_out:
        # Replace NUL characters with an empty string at the line level
        cleaned_lines = (line.replace('\0', '') for line in csv_in)
        reader = csv.reader(cleaned_lines, escapechar='\\')
        writer = csv.writer(csv_out)
        iter = -1  # to account for header
        for row in reader:
            writer.writerow(row)
            iter += 1
            print(iter)


def csv_to_dict(csv_file):
    """Reads a CSV file and converts it into a dictionary, where each key is a
    column name and its value is the list of that column's values across all rows."""
    csv.field_size_limit(2147483647)
    # Initialize logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    start_time = time.time()  # Record the start time
    data_dict = {}

    try:
        with open(csv_file, mode='r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                for key, value in row.items():
                    data_dict.setdefault(key, []).append(value)

        values_count_per_key = {key: len(value) for key, value in data_dict.items()}
        print("Number of values for each key:")
        for key, count in values_count_per_key.items():
            print(f"{key}: {count}")

        end_time = time.time()  # Record the end time
        duration = end_time - start_time  # Calculate the duration
        logger.info(f"Function execution time: {duration} seconds or {duration / 60:.4f} minutes.")
        return data_dict

    except csv.Error as e:
        logger.error(f"WE CHOKED OUT: {e}")


def dict_to_pkl(dictionary, output_pkl_path):
    """ takes a dictionary and saves it as a .pkl"""
    # Save the dictionary to a file
    with open(output_pkl_path, 'wb') as file:
        pickle.dump(dictionary, file)


version_4_1_no_null_path = "<path-to-dataset>/Version 4.1 monthly csv no null/12_December_23_version_4.1.csv"
pkl_path = "<path-to-dataset>/Version 4.2 pkl/12_December_23_version_4.2.pkl"
data_dict = csv_to_dict(version_4_1_no_null_path)

dict_to_pkl(data_dict, pkl_path)