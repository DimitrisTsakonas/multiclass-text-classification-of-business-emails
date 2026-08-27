"""
Purpose:
    Four exploratory attempts at generating document embeddings with
    alternative models: Linq-Embed-Mistral (HuggingFace), Llama 3.1 via
    llama.cpp/GGUF, and Llama 3.1 via Ollama (single-call and batched).
    Plus a standalone token-counting helper for Linq's tokenizer.

Thesis reference:
    None of these were used in the final thesis, which only reports BERT
    and Nomic Embed Text v1.5 (see 4.2.5.4 creator.py) - this file records
    the embedding models explored and abandoned before landing on Nomic.

Inputs:
    A single Version 4.2.5 monthly .pkl file (file_path).

Outputs:
    A .pkl file with an added 'Embeddings' key (naming varies per function
    - see each function's save path).

Notes:
    Several functions here were never fully debugged and would error if
    run - left as-is since this file documents an abandoned exploration
    rather than working code.
"""

import pickle
import os
import time
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from sklearn.preprocessing import normalize
import ollama


def tokens_counter(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    documents = data['To From CC Subject Body']  # List of text documents
    labels = data['Label']

    tokenizer = AutoTokenizer.from_pretrained("Linq-AI-Research/Linq-Embed-Mistral")
    token_counts = [len(tokenizer.encode(doc, truncation=False)) for doc in documents]
    max_tokens = max(token_counts)
    average_tokens = sum(token_counts) / len(token_counts)
    print(f"max tokens: {max_tokens}")
    print(f"avg tokens: {average_tokens}")
    above_4096_count = sum(num > 4096 for num in token_counts)
    above_32768_count = sum(num > 32768 for num in token_counts)
    print(f"instances with tokens above 4096: {above_4096_count}")
    print(f"Instances with tokens above 32768: {above_32768_count}")


# tokens_counter(file_path=r"<path-to-dataset>\Version 4.2.5 c special out\01_January_23_version_4.2.5.pkl")


def linq_embed_mistral_4_2_5_3_creator(file_path):
    start_time = time.time()
    placeholder_embedding = torch.zeros(4096).numpy()  # Placeholder for empty documents
    print(f"placeholder shape: {placeholder_embedding.shape}")

    tokenizer = AutoTokenizer.from_pretrained("Linq-AI-Research/Linq-Embed-Mistral")
    model = AutoModel.from_pretrained("Linq-AI-Research/Linq-Embed-Mistral")
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    print(f"Hidden size of the model: {model.config.hidden_size}")

    def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1  # counts real tokens. gets the index of last real token.
            batch_size = last_hidden_states.shape[0]  # i think batch means, of the tokens we gona pool. not batch docs
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    documents = data['To From CC Subject Body']  # List of text documents
    labels = data['Label']

    document_embeddings = []
    placeholders_used_counter = 0
    for idx, document in enumerate(tqdm(documents,
                                        desc=f"Processing document in {os.path.basename(file_path)}", unit="doc")):

        encoded_input = tokenizer(document, return_tensors='pt', padding=True, truncation=True, max_length=4096)
        encoded_input = {key: value.to(device) for key, value in encoded_input.items()}  # Move to the GPU

        try:
            with torch.no_grad():
                model_output = model(**encoded_input)

            embedding = last_token_pool(model_output.last_hidden_state,
                                        encoded_input['attention_mask'])  # last real token
            embedding = F.normalize(embedding, p=2, dim=1)  # Normalize the embedding with l2 (default)

            # Move to CPU and convert to numpy for saving
            embedding = embedding.cpu().numpy().flatten()
            document_embeddings.append(embedding)  # Append to list
        except Exception as e:
            single_doc_embedding = placeholder_embedding
            placeholders_used_counter += 1
            document_embeddings.append(single_doc_embedding)
            print(f"used placeholder on document with index {idx} due to error: {e}")

    print(f"document_embeddings.shape: {document_embeddings.shape}")
    print(f"data.shape: {data.shape}")
    data['Embeddings'] = document_embeddings

    # save to new file
    new_file_path = file_path.replace("Version 4.2.5 c special out", "Version 4.2.5.3 linq embed mistral").replace(
        "version_4.2.5.pkl", "version_4.2.5.3.pkl")
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)

    with open(new_file_path, 'wb') as file:
        pickle.dump(data, file)

    print(f"place holders used: {placeholders_used_counter}")
    print("Processing completed.")


# linq_embed_mistral_4_2_5_3_creator(
#     file_path=r"<path-to-dataset>\Version 4.2.5 c special out\01_January_23_version_4.2.5.pkl")


def llama3_1_cpp_python_creator(file_path, n_ctx):
    start_time = time.time()

    # Set the path to your GGUF model file
    model_path = r"<path-to-huggingface-cache>\hub\models--bartowski--Meta-Llama-3.1-8B-Instruct-GGUF" \
                 r"\snapshots\bf5b95e96dac0462e2a09145ec66cae9a3f12067\Meta-Llama-3.1-8B-Instruct-Q3_K_S.gguf "

    # Load the model with GPU acceleration (offload all layers)
    llm = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=n_ctx, embedding=True)

    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    documents = data['To From CC Subject Body']  # List of text documents
    document_embeddings = []

    for idx, doc in enumerate(tqdm(documents,
                                   desc=f"Processing document in {os.path.basename(file_path)}", unit="doc")):
        # Generates list of embeddings for each token (num of tokens, 4096)
        token_embeddings = llm.create_embedding(doc)['data'][0]['embedding']
        single_doc_embedding = np.mean(token_embeddings, axis=0)  # Shape: (4096,) mean pooling
        single_doc_embedding = normalize(single_doc_embedding.reshape(1, -1), norm='l2')[0]  # Normalize (L2)
        document_embeddings.append(single_doc_embedding)

    print(f"document_embeddings.shape: {document_embeddings.shape}")
    print(f"original data.shape: {data.shape}")
    data['Embeddings'] = document_embeddings

    # save to new file
    new_file_path = file_path.replace("Version 4.2.5 c special out", f"Version 4.2.5.3 llama 3.1 {n_ctx}").replace(
        "version_4.2.5.pkl", "version_4.2.5.3.pkl")
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)

    with open(new_file_path, 'wb') as file:
        pickle.dump(data, file)

    dur_seconds = time.time() - start_time
    dur_minutes = round(dur_seconds / 60, 1)
    print(f" Time needed: {dur_seconds} seconds or {dur_minutes} minutes.")
    print("Processing completed.")


#
# llama3_1_cpp_python_creator(file_path=r"<path-to-dataset>\Version 4.2.5 c special out\01_January_23_version_4.2.5.pkl",
#                             n_ctx=512)


def llama3_1_ollama_4_2_5_3_creator(file_path):
    start_time = time.time()
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    documents = data['To From CC Subject Body']  # List of text documents

    document_embeddings = [ollama.embeddings(model="llama3.1:latest", prompt=doc)['embedding']
                           for doc in tqdm(documents, desc="Generating embeddings", unit="doc")]

    data['Embeddings'] = document_embeddings

    # save to new file
    new_file_path = file_path.replace("Version 4.2.5 c special out", f"Version 4.2.5.3 ollama 3.1").replace(
        "version_4.2.5.pkl", "version_4.2.5.3.pkl")
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)

    with open(new_file_path, 'wb') as file:
        pickle.dump(data, file)

    dur_seconds = time.time() - start_time
    dur_minutes = round(dur_seconds / 60, 1)
    print(f" Time needed: {dur_seconds} seconds or {dur_minutes} minutes.")
    print("Processing completed.")


# llama3_1_ollama_4_2_5_3_creator(
#     file_path=r"<path-to-dataset>\Version 4.2.5 c special out\02_February_23_version_4.2.5.pkl")


def llama3_1_ollamaembed_4_2_5_3_creator(file_path):
    start_time = time.time()
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    documents = data['To From CC Subject Body']  # List of text documents

    batch_size = 16  # Adjust based on memory limits
    document_embeddings = []
    for i in tqdm(range(0, len(documents), batch_size), desc="Generating embeddings", unit="batch"):
        batch = documents[i:i + batch_size]  # Get a batch of documents
        response = ollama.embed(model="llama3.1:latest", input=batch)  # Pass batch to API
        batch_embeddings = response["embeddings"]  # Extract embeddings (should be a list of lists)
        document_embeddings.extend(batch_embeddings)  # Append to the main list


    data['Embeddings'] = document_embeddings

    # save to new file
    new_file_path = file_path.replace("Version 4.2.5 c special out", f"Version 4.2.5.3 ollama 3.1").replace(
        "version_4.2.5.pkl", "version_4.2.5.3.pkl")
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)

    with open(new_file_path, 'wb') as file:
        pickle.dump(data, file)

    dur_seconds = time.time() - start_time
    dur_minutes = round(dur_seconds / 60, 1)
    print(f" Time needed: {dur_seconds} seconds or {dur_minutes} minutes.")
    print("Processing completed.")


llama3_1_ollamaembed_4_2_5_3_creator(
    file_path=r"<path-to-dataset>\Version 4.2.5 c special out\02_February_23_version_4.2.5.pkl")

