import pandas as pd
import re
import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from tabulate import tabulate
from langdetect import detect, LangDetectException
import math
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


class DataFrameModifier:
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def save_dataframe(self, file_path: str):
        # save the dataframe to a csv file
        self.dataframe.to_csv(file_path, index=False)

    def pretty_printer(self, value: int):
        # print the top 'value' rows of the dataframe in a pretty format
        print(tabulate(self.dataframe.head(value), headers='keys', tablefmt='pretty'))

    def remove_missings(self, column):
        # fill missing values in the specified column with empty strings
        self.dataframe[column] = self.dataframe[column].fillna('')

    def clean_characters(self, column):
        # function to clean characters in a given column
        def clean_text(text):
            # remove non-ASCII characters (incorrectly formatted ones)
            text = str(text)
            text = text.encode('ascii', 'ignore').decode('ascii')

            # use regex to keep only letters, numbers, spaces, and punctuation
            text = re.sub(r"[^a-zA-Z0-9\s.,?!]", "", text)
            return text

        # apply the cleaning function to the specified column with a progress bar
        tqdm.pandas()
        self.dataframe[column] = self.dataframe[column].progress_apply(clean_text)


class DetectLanguagesLangDetect(DataFrameModifier):
    def __init__(self, dataframe: pd.DataFrame):
        super().__init__(dataframe)

    def detect_language(self):
        # function to detect language of a given text
        def detect_language(text):
            try:
                return detect(str(text))
            except LangDetectException:
                print(f"error processing text: {text}")
                return 'unknown'

        # apply language detection to the 'description' column and handle unknown cases
        self.dataframe['language_lang_detect'] = self.dataframe['description'].apply(detect_language)
        mask = self.dataframe['language_lang_detect'] == "unknown"
        self.dataframe.loc[mask, 'language_lang_detect'] = self.dataframe.loc[mask, 'title'].apply(detect_language)

        # count the languages and save them to a file
        language_counts = self.dataframe['language_lang_detect'].value_counts()
        with open("../data/detect_lang_detect.txt", 'w') as f:
            for language, count in language_counts.items():
                f.write(f"{language}: {count}\n")


class DataFrameTranslator(DataFrameModifier):
    def __init__(self, dataframe: pd.DataFrame):
        super().__init__(dataframe)
        model_name = "facebook/m2m100_1.2B"
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        self.model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        self.tgt_lang = 'en'
        self.max_length = 512  # max sequence length
        self.device = 0 if torch.cuda.is_available() else -1
        self.translator = pipeline(
            "translation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )

    def chunk_text(self, text: str) -> list:
        # split the text into chunks that fit within the max length
        tokens = self.tokenizer(text, return_tensors='pt', truncation=False).input_ids[0]
        num_chunks = math.ceil(len(tokens) / int(self.max_length * 0.85))

        chunks = []
        for i in range(num_chunks):
            start_idx = i * int(self.max_length * 0.85)
            end_idx = (i + 1) * int(self.max_length * 0.85)
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)

        return chunks

    def translate_chunk(self, text_chunk: str, src_lang: str, tgt_lang: str) -> str:
        # translate a single chunk of text
        self.tokenizer.src_lang = src_lang
        translation = self.translator(text_chunk, src_lang=src_lang, tgt_lang=tgt_lang, max_length=self.max_length)
        return translation[0]['translation_text']

    def translate_text(self, text: str, src_lang: str, tgt_lang: str) -> str:
        # translate long text by splitting it into chunks and reassembling the translated chunks
        text_chunks = self.chunk_text(text)
        translated_chunks = [self.translate_chunk(chunk, src_lang, tgt_lang) for chunk in text_chunks]
        translated_text = " ".join(translated_chunks)
        return translated_text

    def translate_column(self, column_name: str):
        # translate a whole column of the dataframe
        self.remove_missings(column_name)
        source_texts = self.dataframe[column_name].tolist()
        src_langs = self.dataframe['language_lang_detect'].tolist()

        translated_texts = []
        for text, src_lang in tqdm(zip(source_texts, src_langs), total=len(source_texts), desc="translating"):
            if src_lang == self.tgt_lang:
                translated_texts.append(text)
            else:
                translated_text = self.translate_text(text, src_lang, self.tgt_lang)
                translated_texts.append(translated_text)

        # add the translated texts to a new column in the dataframe
        new_column_name = f"{column_name}_translated"
        self.dataframe[new_column_name] = translated_texts


class SummarizeText(DataFrameModifier):
    def __init__(self, dataframe: pd.DataFrame):
        super().__init__(dataframe)

        model_name = "facebook/bart-large-cnn"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.pipe = pipeline("summarization", model=self.model, tokenizer=self.tokenizer)

    def summarize(self):
        # summarize the 'description_translated' column
        description_list = list(self.dataframe['description_translated'])
        summary = []
        for text in tqdm(description_list):
            output = self.pipe(text, max_length=128, min_length=30, do_sample=False)
            summary.append(output[0]['summary_text'])

        # add the summaries to a new column in the dataframe
        self.dataframe['summary'] = summary
        self.save_dataframe('../data/example_data_summary.csv')
