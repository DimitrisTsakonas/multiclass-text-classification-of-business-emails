"""
Purpose:
    Generates Nomic Embed Text v1.5 document embeddings from Version 4.2.5
    data, via Ollama, in batches of 64 documents.

Thesis reference:
    Experiment 7 - Nomic Embeddings (Section 5.7).

Inputs:
    A single Version 4.2.5 monthly .pkl file (file_path).

Outputs:
    A Version 4.2.5.4 .pkl file with an added 'Embeddings' key.

Notes:
    January-March were run by hand-editing file_path and rerunning
    (as with the other files in this folder); the loop below was added
    afterward to process April-December automatically.
"""

import pickle
import os
import time
from tqdm import tqdm

import ollama


def nomic_embed_text_4_2_5_4_creator(file_path):
    start_time = time.time()
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    documents = data['To From CC Subject Body']  # List of text documents

    batch_size = 64  # Adjust based on memory limits
    document_embeddings = []
    for i in tqdm(range(0, len(documents), batch_size), desc="Generating embeddings", unit="batch"):
        batch = documents[i:i + batch_size]  # Get a batch of documents
        response = ollama.embed(model='nomic-embed-text', input=batch)  # Pass batch to API
        batch_embeddings = response["embeddings"]  # Extract embeddings (should be a list of lists)
        document_embeddings.extend(batch_embeddings)  # Append to the main list

    data['Embeddings'] = document_embeddings

    # save to new file
    new_file_path = file_path.replace("Version 4.2.5 c special out", f"Version 4.2.5.4 nomic embed text").replace(
        "version_4.2.5.pkl", "version_4.2.5.4.pkl")
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)

    with open(new_file_path, 'wb') as file:
        pickle.dump(data, file)

    dur_seconds = time.time() - start_time
    dur_minutes = round(dur_seconds / 60, 1)
    print(f"Finished: {os.path.basename(file_path)} in {dur_seconds:.1f} seconds ({dur_minutes} minutes).")


# === Loop through April to December ===
base_input_dir = r"<path-to-dataset>\Version 4.2.5 c special out"
months = [
    "04_April", "05_May", "06_June", "07_July", "08_August",
    "09_September", "10_October", "11_November", "12_December"
]

for month in months:
    filename = f"{month}_23_version_4.2.5.pkl"
    full_path = os.path.join(base_input_dir, filename)

    if os.path.exists(full_path):
        print(f"Processing: {filename}")
        nomic_embed_text_4_2_5_4_creator(full_path)
    else:
        print(f"File not found: {filename}")