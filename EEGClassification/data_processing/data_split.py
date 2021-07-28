from keras.layers import TextVectorization
from tensorflow import expand_dims
from tensorflow.python.data import Dataset
from tensorflow.python.ops.gen_string_ops import unicode_transcode


def split_dataset(dataset: Dataset, percentage_training: int = 70, percentage_validation: int = 15) -> dict:
    if percentage_training + percentage_validation > 100:
        raise ValueError("Training and validation cannot be greater than 100")

    print("Splitting the dataset. Training:", str(percentage_training) + "%. Validation:", str(
        percentage_validation) + "%. Testing:", str(100 - percentage_validation - percentage_training), "%")

    record_count = len(list(dataset))
    training_count = int((percentage_training / 100) * record_count)
    validation_count = int((percentage_validation / 100) * record_count)

    raw_train_ds = dataset.take(training_count)
    raw_val_ds = dataset.skip(training_count).take(validation_count)
    raw_test_ds = dataset.skip(training_count + validation_count)

    print("Finished splitting the dataset. Training: ", str(len(list(raw_train_ds))), "records. Validation: ", str(
        len(list(raw_val_ds))), "records. Testing: ", str(len(list(raw_test_ds))), "records")
    return {"training": raw_train_ds, "testing": raw_test_ds, "validation": raw_val_ds}


split_dataset.__doc__ = "Splits the dataset into training, testing, and validation sets"


def get_vectorize_layer(dataset: Dataset, max_tokens: int = 10000, parameters: dict = None) -> TextVectorization:
    print("Creating vectorize layer")
    if parameters is not None:
        max_sequence_length = parameters.get("max_sequence_length", 250)
        output_mode = parameters.get("output_mode", "int")
        ngram = parameters.get("ngram", 1)
    else:
        max_sequence_length = 250
        output_mode = "int"
        ngram = 1

    vectorize_layer = TextVectorization(max_tokens=max_tokens, output_mode=output_mode,
                                        output_sequence_length=max_sequence_length, ngrams=ngram)
    text = dataset.map(__clean_text)
    vectorize_layer.adapt(data=text)
    print("Vectorize layer created")
    return vectorize_layer


get_vectorize_layer.__doc__ = "Creates the vectorize layer using the specified dataset"


def vectorize_dataset(dataset: Dataset, vectorize_layer: TextVectorization) -> Dataset:
    print("Vectorizing dataset")
    return dataset.map(lambda text, label: __vectorize_text(vectorize_layer, text, label))


vectorize_dataset.__doc__ = "Standardizes, tokenizes, and vectorizes the dataset for use by AI model"


def __clean_text(text, label):
    cleaned_version_of_text = unicode_transcode(text, "US ASCII", "UTF-8")
    return cleaned_version_of_text


def __vectorize_text(vectorize_layer: TextVectorization, text, label):
    text = expand_dims(input=text, axis=-1)
    return vectorize_layer(text), label
