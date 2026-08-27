"""
Purpose:
    Generates BERT (bert-base-uncased) document embeddings from Version
    4.2.5 data, in two variants: bert_func_4_2_5_1_creator averages
    sentence-level [CLS] embeddings into one document embedding;
    bert_func_4_2_5_2_creator tokenizes the whole document (truncated to
    512 tokens) and averages the token embeddings instead. Unlike the
    earlier 4.2.2.1/4.2.2.2 versions, documents that fail to embed (e.g.
    empty after sentence splitting, or error during tokenization) are kept
    in the dataset with a zero-vector placeholder embedding instead of
    being dropped.

Thesis reference:
    Experiment 4 (Embeddings, Section 5.4) uses the sentence-averaged
    variant; Experiment 6.2 (Token pooling, Section 5.6.2) uses the
    word/token-averaged variant. Built on the corrected Version 4.2.5
    (special characters actually removed, unlike 4.2.3).

Inputs:
    A single Version 4.2.5 monthly .pkl file (file_path).

Outputs:
    A Version 4.2.5.1 or 4.2.5.2 .pkl file (matching whichever function
    ran) with an added 'Embeddings' key.

Notes:
    Only one of the two functions is called at a time (see the bottom of
    the file) - each was run once per month, by hand-editing file_path and
    rerunning, to build up the full year for that embedding variant.
"""

import pickle
from transformers import BertTokenizer, BertModel
import torch
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import os


def bert_func_4_2_5_1_creator(file_path):
    """
    - takes single .pkl filepath as input only
    Takes as input 4.2.5 version of data. Create document embeddings and save them as 4.2.5.1 version that will
    contain:
    note: embeddings are created sentence wise and then the average is kept for the document

    #  'embeddings': embeddings,
    #  'To From CC Subject Body': 'To From CC Subject Body',
    #  'labels': labels

    Process:
    For each document, create embedding for each sentence and then aggregate to get document embedding

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
    placeholder_embedding = torch.zeros(768)  # Placeholder for empty documents
    placeholders_used_counter = 0

    for idx, document in enumerate(tqdm(documents,
                                        desc=f"Processing document in {os.path.basename(file_path)}", unit="doc")):
        # Split document into sentences
        sentences = sent_tokenize(document)

        # Get aggregated document embedding
        sentence_embeddings = []
        for sentence in sentences:
            # 1) Tokenize and encode sentence
            encoded_input = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)
            encoded_input = {key: value.to(device) for key, value in encoded_input.items()}  # Move to the GPU
            # 2) Get BERT embeddings
            with torch.no_grad():
                model_output = model(**encoded_input)
            # 3) Use the [CLS] token's embedding as the sentence embedding
            # cls_embedding shape after .squeeze: 1D array [hidden_size] ie 1D array: [768]
            cls_embedding = model_output.last_hidden_state[:, 0, :].squeeze(0)
            sentence_embeddings.append(cls_embedding)
        if len(sentence_embeddings) == 0:
            aggregated_embedding = placeholder_embedding
            placeholders_used_counter += 1
            print("used placed holder")

        else:
            sentence_embeddings = torch.stack(sentence_embeddings)
            aggregated_embedding = torch.mean(sentence_embeddings, dim=0)  # mean pooling for aggregate

        single_doc_embedding = aggregated_embedding.cpu().numpy()  # MOVE BACK TO CPU and change tensor to NP array
        document_embeddings.append(single_doc_embedding)

    # add key and values into my data dict
    data['Embeddings'] = document_embeddings

    # save to new file
    new_file_path = file_path.replace("Version 4.2.5 c special out", "Version 4.2.5.1 bert sentence").replace(
        "version_4.2.5.pkl", "version_4.2.5.1.pkl")
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)

    with open(new_file_path, 'wb') as file:
        pickle.dump(data, file)

    print(f"place holders used: {placeholders_used_counter}")
    print("Processing completed.")


# Only one of these two is active at a time - comment/uncomment and rerun to switch variants.
# bert_func_4_2_5_1_creator(file_path=r"<path-to-dataset>\Version 4.2.5 c special out\12_December_23_version_4.2.5.pkl")


def bert_func_4_2_5_2_creator(file_path):
    """
    - takes single .pkl filepath as input only
    Takes as input 4.2.5 version of data. Create document embeddings and save them as 4.2.5.2 version that will
    contain:

    note: embeddings are created word wise up to 512 tokens.
    Then the average is kept for the document. Rest is truncated

    #  'embeddings': embeddings,
    #  'To From CC Subject Body': 'To From CC Subject Body',
    #  'labels': labels

    Process:
    For each document, create embedding for each WORD and then aggregate to get document embedding

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
    placeholder_embedding = torch.zeros(768)  # Placeholder for empty documents
    placeholders_used_counter = 0

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

            # Aggregate token embeddings (mean pooling)
            aggregated_embedding = torch.mean(token_embeddings, dim=0)  # Shape: [hidden_size]
            single_doc_embedding = aggregated_embedding.cpu().numpy()  # Move back to CPU and convert to NumPy
            document_embeddings.append(single_doc_embedding)
        except Exception as e:
            single_doc_embedding = placeholder_embedding
            placeholders_used_counter += 1
            document_embeddings.append(single_doc_embedding)
            print(f"used placed holder on document with index {idx} due to error: {e}")

    # add key and values into my data dict
    data['Embeddings'] = document_embeddings

    # save to new file
    new_file_path = file_path.replace("Version 4.2.5 c special out", "Version 4.2.5.2 bert word").replace(
        "version_4.2.5.pkl", "version_4.2.5.2.pkl")
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)

    with open(new_file_path, 'wb') as file:
        pickle.dump(data, file)

    print(f"place holders used: {placeholders_used_counter}")
    print("Processing completed.")


bert_func_4_2_5_2_creator(file_path=r"<path-to-dataset>\Version 4.2.5 c special out\12_December_23_version_4.2.5.pkl")
