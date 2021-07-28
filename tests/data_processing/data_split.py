import unittest
from unittest.mock import patch
import os

from keras.layers import TextVectorization
from tensorflow.keras import preprocessing

from EEGClassification.data_processing import data_split as ds
import tensorflow as tf
import tensorflow_datasets as tfds


class SplitDatasetTest(tf.test.TestCase):
    mock_dataset = None

    def setUp(self):
        super(SplitDatasetTest, self).setUp()
        with tfds.testing.mock_data(num_examples=10):
            self.mock_dataset = tfds.load('ag_news_subset', split='train')

    def test_should_get_correct_percentage_training_records(self):
        result = ds.split_dataset(self.mock_dataset, percentage_training=30)
        self.assertEqual(len(result['training']), 3)

    def test_should_get_correct_percentage_validation_records(self):
        result = ds.split_dataset(self.mock_dataset, percentage_validation=20)
        self.assertEqual(len(result['validation']), 2)

    def test_should_get_correct_percentage_testing_records(self):
        result = ds.split_dataset(self.mock_dataset, percentage_validation=20, percentage_training=50)
        self.assertEqual(len(result['testing']), 3)

    def test_should_fail_if_total_is_greater_than_100_percent(self):
        self.assertRaises(ValueError, ds.split_dataset, self.mock_dataset, percentage_validation=50,
                          percentage_training=80)


class GetTextVectorization(tf.test.TestCase):
    mock_dataset = tfds.load('ag_news_subset', split='train', as_supervised=True)

    def test_should_create_text_vectorization(self):
        result = ds.get_vectorize_layer(self.mock_dataset)
        self.assertIsInstance(result, TextVectorization)


if __name__ == '__main__':
    unittest.main()
