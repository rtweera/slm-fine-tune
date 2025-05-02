"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa

from .nodes import (
    convert_to_dataframe,
    drop_extra_cols,
    split_data,
    tokenize_dataset,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=convert_to_dataframe,
                inputs="sentiments",
                outputs="raw_sentiments",
                name="HF_datasets_convert_to_dataframe_node",
            ),
            node(
                func=drop_extra_cols,
                inputs=["raw_sentiments", "params:dataset_config"],
                outputs="intermediate_sentiments",
                name="drop_extra_cols_node",
            ),
            node(
                func=split_data,
                inputs=["intermediate_sentiments", "params:dataset_config"],
                outputs=[
                    "train_X_sentiments",
                    "validation_X_sentiments",
                    "test_X_sentiments",
                    "train_y_sentiments",
                    "validation_y_sentiments",
                    "test_y_sentiments",
                ],
                name="dataset_split_node",
            ),
            node(
                func=tokenize_dataset,
                inputs=["train_X_sentiments", "tokenizer", "params:tokenizer_config"],
                outputs=["tokenized_sentiments_train_X", 'tokenized_sentiments_readable_train_X'],
                name="tokenize_trainX_node",
            ),
            node(
                func=tokenize_dataset,
                inputs=["validation_X_sentiments", "tokenizer", "params:tokenizer_config"],
                outputs=["tokenized_sentiments_validation_X", 'tokenized_sentiments_readable_validation_X'],
                name="tokenize_validationX_node",
            ),
            node(
                func=tokenize_dataset,
                inputs=["test_X_sentiments", "tokenizer", "params:tokenizer_config"],
                outputs=["tokenized_sentiments_test_X", 'tokenized_sentiments_readable_test_X'],
                name="tokenize_testX_node",
            ),
        ]
    )
