"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.12
"""

import datasets
import pandas as pd

def _merge_datasets(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    # Merge train and test datasets
    merged = pd.concat([train, test], ignore_index=True)
    return merged

def preprocess_dataset(df: datasets.dataset_dict.DatasetDict) -> pd.DataFrame:
    train = df["train"].to_pandas()
    test = df["test"].to_pandas()
    return _merge_datasets(train, test)

def create_model_input_table(df: pd.DataFrame) -> pd.DataFrame:
    # Create a model input table
    model_input_table = df[["text", "label"]].copy()
    return model_input_table