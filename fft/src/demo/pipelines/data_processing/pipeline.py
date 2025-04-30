"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa

from demo.pipelines.data_processing.nodes import preprocess_dataset, create_model_input_table


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=preprocess_dataset,
            inputs="sentiments",
            outputs="preprocessed_sentiments",
            name="preprocess_dataset_node",
        ),
        node(
            func=create_model_input_table,
            inputs="preprocessed_sentiments",
            outputs="model_input_table",
            name="create_model_input_table_node",
        )
    ])
