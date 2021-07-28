import data_processing.main as processing
import model.main as model
import read.main as read

max_features = 10000
parameters = {"max_sequence_length": 250, "output_mode": "int", "ngrams": 5}

dataset = read.load_dataset()
processed_data = processing.process_dataset(dataset=dataset, max_tokens=max_features, parameters=parameters)
model.run(split_dataset=processed_data, vocabulary_size=max_features)
