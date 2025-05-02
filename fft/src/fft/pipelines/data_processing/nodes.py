"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.12
"""

import pandas as pd
import typing as t
import datasets as ds
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split


def convert_to_dataframe(df: ds.DatasetDict) -> pd.DataFrame:
    train = df["train"].to_pandas()
    test = df["test"].to_pandas()
    data = pd.concat([train, test], ignore_index=True)
    return data


def drop_extra_cols(df: pd.DataFrame, parameters: t.Dict) -> pd.DataFrame:
    return df[parameters["keep_columns"]]


def split_data(df: pd.DataFrame, parameters: t.Dict) -> pd.DataFrame:
    X = df[[parameters["text_column"]]]
    y = df[[parameters["target_column"]]]
    inter_X, test_X, inter_y, test_y = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )
    train_X, validation_X, train_y, validation_y = train_test_split(
        inter_X,
        inter_y,
        test_size=parameters["validation_size"],
        random_state=parameters["random_state"],
    )
    data = [train_X, validation_X, test_X, train_y, validation_y, test_y]
    for d in data:
        d.reset_index(drop=True, inplace=True)
    return data


def tokenize_and_get_dataset(
    df: pd.DataFrame, tokenizer: AutoTokenizer, parameters: t.Dict
) -> pd.DataFrame:
    texts = list(df[parameters["tokenize_column"]])
    tokenized = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    # padding to make every text equal - for tokenizing. truncation text length is only upto model context window limit
    df = pd.DataFrame({k: v.numpy().tolist() for k, v in tokenized.items()})
    return df
