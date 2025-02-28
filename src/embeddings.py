import pandas as pd
import numpy as np
from tabulate import tabulate
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


class EmbeddingCalculatror:
    def __init__(self, dataframe: pd.DataFrame):
        # initialize with the dataframe
        self.dataframe = dataframe

    def save_dataframe(self, file_path: str):
        # save the dataframe to a csv file
        self.dataframe.to_csv(file_path, index=False)


class EmbeddingCalculatror(EmbeddingCalculatror):
    def __init__(self, dataframe: pd.DataFrame):
        # initialize the class with the dataframe and set model name
        super().__init__(dataframe)
        self.model_name = "sentence-transformers/all-mpnet-base-v2"

    def calculate_embedding(self, column_name1: str, column_name2: str, data: str, id: str):
        # fill missing values in column2 with values from column1 this is used when the job description is missing
        self.dataframe[column_name2] = self.dataframe[column_name2].fillna(self.dataframe[column_name1])

        # load sentence transformer model
        model = SentenceTransformer(self.model_name)

        # define function to calculate sliding window embeddings
        def sliding_window_embedding(text, model, max_tokens=200, stride=150):
            # split the text into words
            words = text.split()
            embeddings = []

            # process text in chunks using sliding window
            for i in range(0, len(words), stride):
                chunk = ' '.join(words[i:i + max_tokens])
                chunk_embedding = model.encode(chunk)
                embeddings.append(chunk_embedding)

            # combine the embeddings by averaging
            combined_embedding = np.mean(embeddings, axis=0)
            return combined_embedding

        # loop through the columns to calculate embeddings
        for column_name in tqdm([column_name1, column_name2]):
            # calculate sliding window embeddings for each text
            embeddings = self.dataframe[column_name].apply(
                lambda x: sliding_window_embedding(x, model))

            # convert embeddings to a dataframe
            df = pd.DataFrame(embeddings.tolist(), columns=[f'embedding_{i}' for i in range(len(embeddings.tolist()[0]))])

            # add the id column to the dataframe
            df[id] = self.dataframe[id]

            # reorder the columns to ensure 'id' is first
            columns = [id] + [col for col in df.columns if col != id]
            df = df[columns]

            # # display the first few rows of the dataframe
            # print(tabulate(df.head(3), headers='keys', tablefmt='pretty'))

            # save the embeddings dataframe to csv
            df.to_csv(f"../data/{data}_df_embedding_{column_name}.csv", index=False)
