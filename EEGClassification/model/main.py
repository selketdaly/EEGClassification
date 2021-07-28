from .train import *


def run(split_dataset: dict, vocabulary_size: int = 1000, embedding_dimension: int = 16, epochs: int = 10) -> None:
    print("Running model")
    model = create_model(vocabulary_size=vocabulary_size, embedding_dimension=embedding_dimension)
    model = train_model(model=model, training_dataset=split_dataset['training'],
                        validation_dataset=split_dataset['validation'], epochs=epochs)
    accuracy = determine_accuracy(model=model, test_dataset=split_dataset['testing'])
    print("Model has an accuracy of ", str(accuracy * 100), "%")
    plot_accuracy(model=model)
