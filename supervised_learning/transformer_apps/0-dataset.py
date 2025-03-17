#!/usr/bin/env python3
"""
Module to load and prepare dataset for machine translation
"""
import tensorflow as tf
import tensorflow_datasets as tfds
from transformers import BertTokenizerFast


class Dataset:
    """
    Dataset class for machine translation
    """
    def __init__(self):
        """
        Class constructor
        Creates instance attributes
        """
        # Load datasets as supervised
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True)
        
        # Create tokenizers from the training set
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for our dataset
        Args:
            data: tf.data.Dataset whose examples are formatted as a tuple (pt, en)
        Returns:
            tokenizer_pt: Portuguese tokenizer
            tokenizer_en: English tokenizer
        """
        # Initialize the Portuguese tokenizer with pre-trained model
        tokenizer_pt = BertTokenizerFast.from_pretrained(
            'neuralmind/bert-base-portuguese-cased')
        
        # Initialize the English tokenizer with pre-trained model
        tokenizer_en = BertTokenizerFast.from_pretrained(
            'bert-base-uncased')
        
        # Train the tokenizers with maximum vocabulary size of 2^13
        vocab_size = 2**13
        
        return tokenizer_pt, tokenizer_en
