"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.12
"""

import pandas as pd
import typing as t
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict


def convert_to_dataframe(df: DatasetDict) -> pd.DataFrame:
    train = df["train"].to_pandas()
    test = df["test"].to_pandas()
    data = pd.concat([train, test], ignore_index=True)
    return data


def drop_extra_cols(df: pd.DataFrame, parameters: t.Dict) -> pd.DataFrame:
    return df[parameters["keep_columns"]]


def split_data(df: pd.DataFrame, parameters: t.Dict) -> pd.DataFrame:
    inter, test = train_test_split(
        df, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )
    train, validation = train_test_split(
        inter,
        test_size=parameters["validation_size"],
        random_state=parameters["random_state"],
    )
    data = [train, validation, test]
    for d in data:
        d.reset_index(drop=True, inplace=True)
    return data


def _tokenize_function(
    data: Dataset, tokenizer: AutoTokenizer, parameters: t.Dict
) -> Dataset:
    # padding to make every text equal - for tokenizing. truncation text length is only upto model context window limit
    tokenized: Dataset =  tokenizer(
        data[parameters['tokenize_column']], padding=parameters["padding"], truncation=parameters["truncation"]
    )
    # Add the non altered columns back to the dataset
    # for key in data:
    #     if key not in tokenized:
    #         tokenized[key] = data[key]
    return tokenized[[parameters['keep_columns']]]


def tokenize_dataset(
    df: pd.DataFrame, tokenizer: AutoTokenizer, parameters: t.Dict
) -> t.Tuple[Dataset, pd.DataFrame]:
    data = Dataset.from_pandas(df)
    tokenized: Dataset = data.map(
        _tokenize_function,
        batched=parameters["batched"],
        batch_size=parameters["batch_size"],
        fn_kwargs={"tokenizer": tokenizer, "parameters": parameters},
    )
    df = tokenized.to_pandas()
    return [tokenized, df]
