import os

from tensorflow.python.data import Dataset
from tensorflow.python.keras.preprocessing.text_dataset import text_dataset_from_directory

from .file_extraction import extract_tars

directory = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # Gets the root of the project
dataset_directory = directory + "/data"


def load_dataset() -> Dataset:
    print("Loading the dataset")
    extract_tars(directory, dataset_directory)
    return text_dataset_from_directory(dataset_directory, batch_size=32)


load_dataset.__doc__ = "Loads the dataset for use by AI model"
