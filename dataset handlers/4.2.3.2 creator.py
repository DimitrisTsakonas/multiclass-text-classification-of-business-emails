"""
Purpose:
    Generates BERT (bert-base-uncased) document embeddings from Version
    4.2.3 data: tokenizes the whole document (truncated to 512 tokens) and
    keeps only the [CLS] token's embedding as the document representation.

Thesis reference:
    Experiment 6.1 - Document CLS (Section 5.6.1). Built on Version 4.2.3,
    which - despite its name - never actually removed special characters
    (see 4.2.3 creator.py), so these embeddings were generated on text that
    still contains them.

Inputs:
    A single Version 4.2.3 monthly .pkl file (file_path).

Outputs:
    A Version 4.2.3.2 .pkl file with an added 'Embeddings' key, plus a
    '..._skipped.pkl' file listing any documents skipped due to errors.
"""

import pickle
from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm
import os


def bert_func_4_2_3_2_creator(file_path):
    """
    - takes single .pkl filepath as input only
    Takes as input 4.2.3 version of data. Create document embeddings and save them as 4.2.3.2 version that will
    contain:

    note: embeddings are created word wise up to 512 tokens.
   cls is kept. Rest is truncated

    #  'embeddings': embeddings,
    #  'To From CC Subject Body': 'To From CC Subject Body',
    #  'labels': labels

    Process:
    For each document, create document embedding using cls token

    """

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Move the model to the selected device (GPU or CPU)
    model = model.to(device)

    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    documents = data['To From CC Subject Body']  # List of text documents
    labels = data['Label']

    document_embeddings = []
    skipped_docs_counter = 0
    skipped_docs_list = []
    skipped_docs_indices = []  # Store indices of skipped documents

    for idx, document in enumerate(tqdm(documents,
                                        desc=f"Processing document in {os.path.basename(file_path)}", unit="doc")):
        # Tokenize the document
        encoded_input = tokenizer(document, return_tensors='pt', padding=True, truncation=True, max_length=512)
        encoded_input = {key: value.to(device) for key, value in encoded_input.items()}  # Move to the GPU

        try:
            with torch.no_grad():
                model_output = model(**encoded_input)

            # Extract token embeddings (excluding [CLS] and [SEP] if present)
            token_embeddings = model_output.last_hidden_state.squeeze(0)  # Shape: [seq_length, hidden_size]

            # Extract the embedding for the [CLS] token (index 0)
            cls_embedding = token_embeddings[0]  # Shape: [hidden_size]
            single_doc_embedding = cls_embedding.cpu().numpy()  # Move back to CPU and convert to NumPy
            document_embeddings.append(single_doc_embedding)
        except Exception as e:
            skipped_docs_counter += 1
            skipped_docs_list.append(document)
            skipped_docs_indices.append(idx)  # Store indices of skipped documents
            print(f"Skipped document with index {idx} due to error: {e}")

    # Remove skipped documents from data to make sure embedding list is same size and order as data
    data['To From CC Subject Body'] = [doc for i, doc in enumerate(documents) if i not in skipped_docs_indices]
    data['Label'] = [label for i, label in enumerate(labels) if i not in skipped_docs_indices]

    # add key and values into my data dict
    data['Embeddings'] = document_embeddings

    # save to new file
    new_file_path = file_path.replace("Version 4.2.3 c special out", "Version 4.2.3.2 bert cls").replace(
        "version_4.2.3.pkl", "version_4.2.3.2.pkl")
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)

    with open(new_file_path, 'wb') as file:
        pickle.dump(data, file)

    if len(skipped_docs_list) != 0:
        skipped_docs_path = file_path.replace("Version 4.2.3 c special out", "Version 4.2.3.2 bert cls").replace(
            "version_4.2.3.pkl", "version_4.2.3.2_skipped.pkl")

        with open(skipped_docs_path, 'wb') as file:
            pickle.dump(skipped_docs_list, file)

    print(f"len skipped_docs_list: {len(skipped_docs_list)}")
    print(f"skipped_docs_counter: {skipped_docs_counter}")
    print(f"indices of doc skipped: {skipped_docs_indices}")
    print("Processing completed.")


bert_func_4_2_3_2_creator(
    file_path=r"<path-to-dataset>\Version 4.2.3 c special out\12_December_23_version_4.2.3.pkl")