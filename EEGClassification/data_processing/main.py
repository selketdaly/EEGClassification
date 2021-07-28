import os

from tensorflow import data

from .data_split import split_dataset, vectorize_dataset, get_vectorize_layer

directory = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
dataset_directory = directory + "/data"


def process_dataset(dataset: data.Dataset, max_tokens: int = 10000, parameters: dict = None) -> dict:
    print("Processing dataset")
    split_data = split_dataset(dataset=dataset)
    vectorize_layer = get_vectorize_layer(dataset=split_data['training'], max_tokens=max_tokens, parameters=parameters)
    return {"training": vectorize_dataset(dataset=split_data['training'], vectorize_layer=vectorize_layer),
            "testing": vectorize_dataset(dataset=split_data['testing'], vectorize_layer=vectorize_layer),
            "validation": vectorize_dataset(dataset=split_data['validation'], vectorize_layer=vectorize_layer)}


process_dataset.__doc__ = "Processes the dataset for use by AI model"
