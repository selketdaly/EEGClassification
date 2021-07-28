import matplotlib.pyplot as plt
from keras.callbacks import History
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Dropout, GlobalAveragePooling1D, Dense
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.python.data import Dataset

history = History()


def create_model(vocabulary_size: int = 10000, embedding_dimension: int = 16) -> Sequential:
    print("Creating model for training")
    model = Sequential(layers=[Embedding(input_dim=vocabulary_size + 1, output_dim=embedding_dimension), Dropout(0.2),
                               GlobalAveragePooling1D(), Dropout(0.2), Dense(1)])
    model.compile(loss=BinaryCrossentropy(from_logits=True), optimizer='adam',
                  metrics=[BinaryAccuracy(threshold=0.0)])
    print("Created model")
    return model


create_model.__doc__ = "Creates a model which can be trained"


def train_model(model: Sequential, training_dataset: Dataset, validation_dataset: Dataset,
                epochs: int = 10) -> Sequential:
    print("Starting model training")
    model.fit(x=training_dataset, validation_data=validation_dataset, epochs=epochs, callbacks=[history])
    print("Completed model training")
    return model


train_model.__doc__ = "Trains the given model with the training dataset, and validates the result"


def determine_accuracy(model: Sequential, test_dataset: Dataset) -> int:
    print("Evaluating model")
    loss, accuracy, *other = model.evaluate(x=test_dataset)
    print("Model evaluated")
    return accuracy


determine_accuracy.__doc__ = "Uses the test dataset to determine the accuracy of the model"


def plot_accuracy() -> None:
    history_dictionary = history.history
    accuracy = history_dictionary['binary_accuracy']
    validation_accuracy = history_dictionary['val_binary_accuracy']

    epochs = range(1, len(accuracy) + 1)

    plt.plot(epochs, accuracy, 'bo', label='Training acc')
    plt.plot(epochs, validation_accuracy, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    plt.show()


plot_accuracy.__doc__ = "Plots the accuracy of the model"
