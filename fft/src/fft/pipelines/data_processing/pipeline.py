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
                    "train_sentiments",
                    "validation_sentiments",
                    "test_sentiments",
                ],
                name="dataset_split_node",
            ),
            node(
                func=tokenize_dataset,
                inputs=["train_sentiments", "tokenizer", "params:tokenizer_config"],
                outputs=[
                    "tokenized_sentiments_train",
                    "tokenized_sentiments_readable_train",
                ],
                name="tokenize_train_node",
            ),
            node(
                func=tokenize_dataset,
                inputs=[
                    "validation_sentiments",
                    "tokenizer",
                    "params:tokenizer_config",
                ],
                outputs=[
                    "tokenized_sentiments_validation",
                    "tokenized_sentiments_readable_validation",
                ],
                name="tokenize_validation_node",
            ),
            node(
                func=tokenize_dataset,
                inputs=["test_sentiments", "tokenizer", "params:tokenizer_config"],
                outputs=[
                    "tokenized_sentiments_test",
                    "tokenized_sentiments_readable_test",
                ],
                name="tokenize_test_node",
            ),
        ]
    )
