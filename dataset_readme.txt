
Dataset Pipeline
-------
Step 1: Downloaded emails as emls and some html. (Version 1)
Step 2: Converted html emails into .txt. (Version 2)
Step 3: Save emails into .csv format (called data.csv) using function csv_creator.py (parse_email() and save_to_csv()) (Version 3)
Step 4: Clean csv from null values and save them in a new .csv called data_no_null.csv. Used nul_removal() from csv_handler.py (Version 3.1)
Step 5: Loaded data_no_null.csv into a dictionary and saved it into a .pkl file (called no_null_dict.pkl) using csv_to_dict() from csv_handler.py (Version 3.2)

Step 6: Save emails into .csv format (called data.csv) using function csv_creator.py (parse_email() and save_to_csv()) (Version 4)
Step 7: Clean csv from null values and save them in a new no-null csv. Used nul_removal() from csv_handler.py (Version 4.1)
Step 8: Loaded the no-null csv into a dictionary and saved it into a .pkl file using csv_to_dict() and dict_to_pkl() from csv_handler.py (Version 4.2)
Step 9: Converts Version 4.2 into Version 4.2.1: removes class C, keeps text+label only, encodes labels numerically, lowercases and stems the text (for TF-IDF). Used dataset handlers/4.2.1 creator.py. (Version 4.2.1)
Step 10: Converts Version 4.2 into Version 4.2.2: same as above but without lowercasing/stemming, to feed the BERT embedding generators. Used dataset handlers/4.2.2 creator.py. (Version 4.2.2)
Step 11: Generates BERT (bert-base-uncased) document embeddings from Version 4.2.2, in two variants: sentence-averaged [CLS] embeddings (Version 4.2.2.1) and whole-document token-averaged embeddings (Version 4.2.2.2). Used dataset handlers/4.2.2.1 and 4.2.2.2 creator.py.
Step 12: Converts Version 4.2 into Version 4.2.3: removes class C, keeps text+label, encodes labels numerically. Intended as the special-character-removed input, though special-character removal doesn't actually happen due to a bug - kept as-is, fixed in Version 4.2.5. Used dataset handlers/4.2.3 creator.py. (Version 4.2.3)
Step 13: Generates BERT document embeddings from Version 4.2.3 using the whole-document [CLS] token (Experiment 6.1 - Document CLS). Used dataset handlers/4.2.3.2 creator.py. (Version 4.2.3.2)
Step 14: Converts Version 4.2 into Version 4.2.4: removes class C, keeps text+label, encodes labels numerically, removes special characters, lowercases and stems. Used dataset handlers/4.2.4 creator.py. (Version 4.2.4)
Step 15: Converts Version 4.2 into Version 4.2.5: same as 4.2.4 but without lowercasing/stemming, correctly removing special characters this time. Used dataset handlers/4.2.5 creator.py. (Version 4.2.5)
Step 16: Generates BERT document embeddings from Version 4.2.5, in two variants: sentence-averaged [CLS] embeddings (Version 4.2.5.1) and whole-document token-averaged embeddings (Version 4.2.5.2). Used dataset handlers/4.2.5.1 and 4.2.5.2 creator.py.
Step 17: Four exploratory attempts at generating document embeddings with alternative models (Linq-Embed-Mistral, Llama 3.1 via llama.cpp, Llama 3.1 via Ollama) from Version 4.2.5 data - never used in the final thesis. Used dataset handlers/4.2.5.3 creator.py. (Version 4.2.5.3)
Step 18: Generates Nomic Embed Text v1.5 document embeddings from Version 4.2.5 data via Ollama (Experiment 7 - Nomic Embeddings). Used dataset handlers/4.2.5.4 creator.py. (Version 4.2.5.4)
--------