"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.12
"""

import logging
import typing as t
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

def split_dataset(df: pd.DataFrame, parameters: t.Dict) -> t.Tuple:
    X = df[parameters['features']]
    y = df[parameters['target']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=parameters['test_size'], random_state=parameters['random_state'])
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    return X_train, X_test, y_train, y_test


def train_model(train_X: pd.DataFrame, train_y: pd.Series) -> LogisticRegression:
    model = LogisticRegression()
    model.fit(train_X, train_y)
    return model


def evaluate_model(model: LogisticRegression, test_X: pd.DataFrame, test_y: pd.Series) -> float:
    ...