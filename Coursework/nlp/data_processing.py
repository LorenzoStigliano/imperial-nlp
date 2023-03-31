import os
import re
from typing import Optional, Tuple
from time import perf_counter

from .utils import Utils
from .config import DATA_FOLDER, DATA_PCL_NAME, DATA_CATEGORIES_NAME, TRAIN_ID, DEV_ID

import ast
import contractions
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
import nlpaug.augmenter.word as naw
import nlpaug.flow as naf
from transformers import BertTokenizer
from nltk.stem import PorterStemmer
from autocorrect import Speller


class DataProcessing(Utils):
    def __init__(self) -> None:
        # Preprocessing objects and parameters
        self.spell = Speller(lang="en")
        self.porter = PorterStemmer()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.augmentor = naf.Sometimes([
            naw.SynonymAug(aug_src='wordnet'),
            naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="insert"),
            naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute"),
        ])

        # Model Tokenizer parameters
        self.MAX_SEQ_LEN = 512
        self.PAD_INDEX = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        self.UNK_INDEX = self.tokenizer.convert_tokens_to_ids(self.tokenizer.unk_token)

        # Paths
        self.data_folder = DATA_FOLDER
        self.create_folder(self.data_folder)
        self.data_pcl_name = DATA_PCL_NAME
        self.data_categories_name = DATA_CATEGORIES_NAME
        self.train_id_path = os.path.join(self.data_folder, TRAIN_ID)
        self.dev_id_path = os.path.join(self.data_folder, DEV_ID)
        self.data_pcl_path = os.path.join(self.data_folder, self.data_pcl_name)
        self.data_pcl_path_train = os.path.join(
            self.data_folder, "train_" + self.data_pcl_name
        )
        self.data_pcl_path_dev = os.path.join(
            self.data_folder, "dev_" + self.data_pcl_name
        )
        self.data_categories_path = os.path.join(
            self.data_folder, self.data_categories_name
        )
        self.preprocessed_data_pcl_path = os.path.join(
            self.data_folder, f"preprocessed_{self.data_pcl_name}"
        )
        self.preprocessed_data_pcl_path_train = os.path.join(
            self.data_folder, f"train_preprocessed_{self.data_pcl_name}"
        )
        self.preprocessed_data_pcl_path_dev = os.path.join(
            self.data_folder, f"dev_preprocessed_{self.data_pcl_name}"
        )
        self.preprocessed_data_categories_path = os.path.join(
            self.data_folder, f"preprocessed_{self.data_categories_name}"
        )
        # Datasets
        self.initial_data_extraction()  # tsv to csv
        self.data_pcl = self.load_data(self.data_pcl_path).dropna()
        self.data_pcl_train = None
        self.data_pcl_dev = None
        self.preprocessed_data_pcl = self.load_data(self.preprocessed_data_pcl_path)
        self.preprocessed_data_pcl_train = None
        self.preprocessed_data_pcl_dev = None
        self.data_categories = self.load_data(self.data_categories_path)
        self.preprocessed_data_categories = self.load_data(
            self.preprocessed_data_categories_path
        )
        # Split train/dev
        self.split_train_dev()

    def __repr__(self) -> str:
        return "%s(%r)" % (self.__class__, self.__dict__)

    def __str__(self) -> str:
        return "%s(%r)" % (self.__class__, self.__dict__)

    def initial_data_extraction(self) -> None:
        """
        Transforms the initial TSV files into CSV format if the dataset is not already stored in the machine.
        """
        if not (os.path.exists(self.data_pcl_path)) or not (
            os.path.exists(self.data_categories_path)
        ):
            print("Convert the initial tsv files into csv...")

            # Transform the initial tsv files into csv
            def preprocess_line(line: str) -> list:
                return line[:-1].split("\t")

            # PCL
            with open(self.data_pcl_path[:-4] + ".tsv", "r+") as f:
                dataset = f.readlines()
                dataset = dataset[4:]  # drop first warning lines
                dataset = list(map(preprocess_line, dataset))
            df = pd.DataFrame(dataset)
            df.columns = [
                "par_id",
                "art_id",
                "keyword",
                "country_code",
                "text",
                "label",
            ]
            # process par_id and art_id
            df["par_id"] = df["par_id"].apply(lambda x: int(x))
            df["art_id"] = df["art_id"].apply(lambda x: int(x[2:]))
            # add binary label like the paper
            df["label"] = df["label"].apply(
                lambda x: np.nan if x == "" else int(x)
            )
            df["binary_label"] = df["label"].apply(
                lambda x: 1
                if not (np.isnan(x)) and x >= 2
                else 0
                if not (np.isnan(x))
                else np.nan
            )
            df.dropna(inplace=True)  # ~one of two rows
            # Convert to float (not long!) for PyTorch
            df["label"] = df["label"].astype(np.float16)
            df["binary_label"] = df["binary_label"].astype(np.float16)
            df.to_csv(self.data_pcl_path, index=False)

            # Categories
            with open(self.data_categories_path[:-4] + ".tsv", "r+") as f:
                dataset = f.readlines()
                dataset = dataset[4:]  # drop first warning lines
                dataset = list(map(preprocess_line, dataset))
            df = pd.DataFrame(dataset)
            df.columns = [
                "par_id",
                "art_id",
                "text",
                "keyword",
                "country_code",
                "span_start",
                "span_finish",
                "span_text",
                "pcl_category",
                "number_of_annotators",
            ]
            # process par_id and art_id
            df["par_id"] = df["par_id"].apply(lambda x: int(x))
            df["art_id"] = df["art_id"].apply(lambda x: int(x[2:]))

            df.dropna(inplace=True)  # ~one of two rows
            df.to_csv(self.data_categories_path, index=False)

    def load_data(self, path: str) -> Optional[pd.DataFrame]:
        """
        :param path: path to CSV file

        Read CSV file
        """
        if os.path.exists(path):
            df = pd.read_csv(path)
            # convert back the lists saved as strings
            for c in ["input_ids", "attention_mask"]:
                if c in df.columns:
                    df[c] = df[c].apply(lambda x: ast.literal_eval(x))
            return df
        return None

    def save_data(self, df: pd.DataFrame, type: str = "pcl") -> None:
        """
        :param df: data frame object to save.
        :param type: 'pcl' or 'categories'.

        Transforms DataFrame object to CSV and updates the relevant variables. Saves the CSV to a local file.
        """
        if type == "pcl":
            self.preprocessed_data_pcl = df
            df.to_csv(self.preprocessed_data_pcl_path, index=False)
        elif type == "categories":
            self.preprocessed_data_categories = df
            df.to_csv(self.preprocessed_data_categories_path, index=False)
        else:
            raise ValueError(f"Unknown type: {type}")

    # Spell checker
    def autospell(self, text: str) -> str:
        spells = [self.spell(w) for w in text.split()]
        return " ".join(spells)

    # Spell checker, normalisation of punctuation, white-space normalisation
    # removing numbers, removing contractions, replacing the repetitions of punctations
    def clean_text(self, text: str, whitespacing = True, standard_tokens = True, punctuation=True, stop_words = False) -> str:

        stop_words = ['this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
                      'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'don', "don't", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'shan', "shan't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

        if whitespacing:
            # Whtespacing and other standarisation
            text = text.strip('"')  # removing " at start of sentences
            text = re.sub('[ ]*', ' ', text)
            text = re.sub('<h>', '.', text)
        if standard_tokens:
            # Removing unecessary info
            text = re.sub("(?<![\w])[0-9]+[.,]?[0-9]*(?![\w])", "[NUM]", text)
            text = re.sub("\[NUM\]-\[NUM\]", "[NUM]", text)
            # Again to delete account numbers lol 12-5223-231
            text = re.sub("\[NUM\]-\[NUM\]", "[NUM]", text)
            text = re.sub(r"https? : \S+", "[WEBSITE]", text)  # Tokenize links
            text = re.sub("(?<![\w])20[0-5][0-9]-?[0-9]*", "[YEAR]", text)  # Year token
            text = re.sub(r"@[\S]+", "[USERNAME]", text)  # removing referencing on usernames with @
            text = re.sub("(?<![\w])1[0-9]{3}-?[0-9]*", "[YEAR]", text)  # Year token
            #text = re.sub("(?<=\[NUM\])-(?=[a-zA-Z])", " ", text)
        if punctuation:
            text = re.sub(r":\S+", "", text)  # removing smileys with : (like :),:D,:( etc)
            text = re.sub(r"\"+", "", text)  # replacing repetitions of punctations
            text = re.sub(r"(\W)(?=\1)", "", text)  # replacing repetitions of punctations
        if stop_words:
            # Stop words --> token
            for stop in stop_words:
                text = re.sub(stop, '[STOPWORD]', text)

        return text.strip()
    
    def stopwords(self, text: str) -> str:

        stop_words = ['this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                      'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 
                      'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 
                      'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 
                      'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'don', "don't", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 
                      'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'shan', "shan't", 'wasn', "wasn't", 'weren', 
                      "weren't", 'won', "won't", 'wouldn', "wouldn't"]

        for stop in stop_words:
            text = re.sub(stop, '[STOPWORD]', text)

        return text.strip()




    # Tokenize with pre-trained model
    def data_process(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        # Enocode the text
        tokens = self.tokenizer(
            df.tolist(), truncation=True, max_length=self.MAX_SEQ_LEN, padding=True
        )
        # Encode the tokens and pad the text output
        input_ids = tokens["input_ids"]
        # Retrieve the attention masks
        attention_mask = tokens["attention_mask"]
        return input_ids, attention_mask

    def data_augmentation_class_rebalance(self, df: pd.DataFrame, augmentor, text_col: str = "text", label_col: str = "binary_label") -> pd.DataFrame:
        all_data = [df]
        n = int(len(df[df[label_col] == 0]) / len(df[df[label_col] == 1])) if len(df[df[label_col] == 1]) != 0 else 0
        n = n // 2  # otherwise we rebalance too much
        print(f"Data augmentation: rebalancing {n} times...")
        for i in range(n):
            print(f"    Iteration {i}")
            top = perf_counter()
            df_new = df[df[label_col] == 1].copy(deep=True)
            texts = df_new[text_col].tolist()
            augmented_text = [augmentor.augment(text)[0] for text in texts]
            df_new[text_col] = augmented_text
            all_data.append(df_new)
            print(f"Done in {int(perf_counter() - top)}s")

        return pd.concat(all_data, axis=0)

    def run_preprocessing_pcl(self) -> pd.DataFrame:
        """
        Returns preprocessed PCL data. If it hasn't been preprocessed yet, it does so first.
        """
        if self.preprocessed_data_pcl is not None:
            return self.preprocessed_data_pcl
        print("Preprocessing PCL data...")
        df = self.data_pcl
        # PREPROCESS THE DATA ELSE
        df_process = df.dropna()

        # Stem keyword
        df_process["keyword"] = df_process["keyword"].apply(
            lambda word: self.porter.stem(word)
        )

        # Convert countries to categtorical label value
        df_process["country_code"] = pd.Categorical(
            df_process["country_code"], categories=df_process["country_code"].unique()
        ).codes

        # Spell checker, normalisation of punctuation,
        # white-space normalisation,
        # removing numbers, removing contractions,
        # replacing the repetitions of punctations
        # extending the dataset with the column clean_text
        df_process["clean_text"] = df_process["text"].apply(
            lambda x: self.clean_text(x)
        )

        # Data augmentation
        df_process = self.data_augmentation_class_rebalance(df_process, self.augmentor, text_col="clean_text", label_col="binary_label")

        # Tokenize with pre-trained model
        # extending the dataset with the column clean_text
        input_ids, attention_mask = self.data_process(df_process["clean_text"])
        df_process["input_ids"] = input_ids
        df_process["attention_mask"] = attention_mask

        # Save the preprocessed dataset
        self.preprocessed_data_pcl = df_process
        self.save_data(df_process, type="pcl")
        return df_process

    # We don't process categories as it is not going to be used
    """
    def run_preprocessing_categories(self) -> pd.DataFrame:
        # Returns preprocessed categories data.
        # If it hasn't been preprocessed yet, it does so first.
        if self.preprocessed_data_categories is not None:
            return self.preprocessed_data_categories
        df = self.data_categories
        # PREPROCESS THE DATA ELSE

        df_process = df
        # Save the preprocessed dataset

        # self.save_data(df_process, type="categories")
        return df_process
    """

    def split_train_dev(self) -> None:
        """
        Loads the raw/preprocessed train/test sets, and initializes all relevant variables.
        """
        print("Splitting train/dev sets...")
        df = self.data_pcl
        if self.preprocessed_data_pcl is None:
            print("First, preprocessing the data...")
            df_process = self.run_preprocessing_pcl()
            print("Splitting...")
        df_process = self.preprocessed_data_pcl

        train_id = pd.read_csv(self.train_id_path)
        dev_id = pd.read_csv(self.train_id_path)

        self.data_pcl_train = df[df["par_id"].isin(train_id["par_id"].tolist())]
        self.data_pcl_dev = df[df["par_id"].isin(dev_id["par_id"].tolist())]
        self.preprocessed_data_pcl_train = df_process[
            df_process["par_id"].isin(train_id["par_id"].tolist())
        ]
        self.preprocessed_data_pcl_dev = df_process[
            df_process["par_id"].isin(dev_id["par_id"].tolist())
        ]
