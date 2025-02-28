import os
import pandas as pd
import numpy as np
import faiss
from preprocess_data import DataFrameTranslator, DetectLanguagesLangDetect, SummarizeText
from embeddings import EmbeddingCalculatror

# constants for column names
LABEL = "label"
TITLE = "title"
DESCRIPTION = "description"

###########################################################
############## preprocess data description  ###############
###########################################################

# calculate embeddings for the data description
data_description_filepath = "../data/data_description_df_embedding_description.csv"
if not os.path.exists(data_description_filepath):
    print("### preprocessing data description ###")
    data_description_path = "../data/data_description.csv"
    data_description = pd.read_csv(data_description_path)

    # generate embeddings using embeddingcalculatror class
    embedding_calculator = EmbeddingCalculatror(data_description)
    embedding_calculator.calculate_embedding(LABEL, DESCRIPTION, "data_description", 'code')

###########################################################
################# detect language #########################
###########################################################
print("### detecting language ###")

example_data_path = "../data/example_data.csv"
example_data = pd.read_csv(example_data_path)

# select example data (rows 1:3)
example_data = example_data.iloc[[1]]

# language detection
language_detector = DetectLanguagesLangDetect(example_data)
language_detector.detect_language()  # detects language for title, description

###########################################################
############## translate to english ######################
###########################################################
print("### translating to english ###")

# initialize translator and translate columns
translator = DataFrameTranslator(example_data)
translator.translate_column(TITLE)
translator.translate_column(DESCRIPTION)

###########################################################
############## generate relevant text ####################
###########################################################
print("### generating relevant text ###")

# initialize summarizer and summarize description
summarizer = SummarizeText(example_data)
summarizer.summarize()

###########################################################
############## calculate sentence embeddings ##############
###########################################################
print("### calculating sentence embeddings ###")

embedding_calculator = EmbeddingCalculatror(example_data)
embedding_calculator.calculate_embedding("title_translated", "summary", "example_data", "id")

###########################################################
############## compare embeddings #########################
###########################################################

# load the embedding data
embedding_title_query = pd.read_csv("../data/example_data_df_embedding_title_translated.csv")
embedding_description_query = pd.read_csv("../data/example_data_df_embedding_summary.csv")
embedding_label_database = pd.read_csv("../data/data_description_df_embedding_label.csv")
embedding_description_database = pd.read_csv("../data/data_description_df_embedding_description.csv")

# select a single row from the query data
embedding_title_query = embedding_title_query.head(1)
embedding_description_query = embedding_description_query.head(1)


# extract embeddings (excluding 'code' column)
def extract_embeddings(df):
    return df.iloc[:, 1:].values  # exclude 'code' column


db_codes = embedding_label_database['code'].values  # extract 'code' column

# query embeddings
embedding_title_query = embedding_title_query.sample(1)
title_query = extract_embeddings(embedding_title_query)
description_query = extract_embeddings(embedding_description_query)

# database embeddings
title_db = extract_embeddings(embedding_label_database)
description_db = extract_embeddings(embedding_description_database)

# weight the embeddings (70% title, 30% description)
weighted_query = 0.7 * title_query + 0.3 * description_query
weighted_db = 0.7 * title_db + 0.3 * description_db


# normalize embeddings
def normalize(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + 1e-10)  # avoid division by zero


weighted_query = normalize(weighted_query)
weighted_db = normalize(weighted_db)

# faiss index for similarity search
dim = weighted_db.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(weighted_db.astype(np.float32))

# search for top 5 most similar entries
distance, indices = index.search(weighted_query.astype(np.float32), 5)

# retrieve the top 5 matching codes
top_5_codes = db_codes[indices[0]]

# fetch relevant details for the top 5 codes
data_description = pd.read_csv("../data/data_description.csv")
labels, uris = [], []

for code in top_5_codes:
    temp_row = data_description[data_description["code"] == code]
    labels.append(temp_row["label"].values[0])
    uris.append(temp_row["isco_uri"].values[0])

# create a dataframe with the results
prediction_df = pd.DataFrame({"label": labels, "isco_uri": uris})
print("done")

# save results and query data to csv
prediction_df.to_csv("../data/result.csv", index=False)
example_data.to_csv("../data/query.csv", index=False)
