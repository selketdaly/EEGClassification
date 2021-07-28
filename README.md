# EEG Classification
Trains, validates and tests a model to classify EEG text reports.

1. Provide the text data:
   * If data is in a tar file, add the tar file to the project root. The application will extract the data from any tars that have not already been extracted
   * If the data is not in a tar file, add the data to the `/data` directory in the project root. If the folder does not exist, create the folder before adding the data
2. Ensure the project dependencies have been installed (see `requirements.txt`)
3. Run `EEGClassification\main.py`
4. The script will create and train a model using the data, and then test it to determine accuracy. A graph will be provided to demonstrate accuracy over time

The following variables can be modified to test different models:
* max_sequence_length
* output_mode
* ngrams

If these are not provided, defaults will be used. These can be set in the `parameters` dict in `EEGClassification\main.py`