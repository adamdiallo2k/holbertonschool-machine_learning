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
        self.data_train = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='train',
            as_supervised=True
        )
        self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='validation',
            as_supervised=True
        )

        # Create tokenizers (pretrained) from Hugging Face
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        """
        Loads pretrained sub-word tokenizers for our dataset.

        Args:
            data: tf.data.Dataset whose examples are formatted as (pt, en)

        Returns:
            tokenizer_pt: Portuguese tokenizer
            tokenizer_en: English tokenizer
        """
        # Portuguese tokenizer from neuralmind
        tokenizer_pt = BertTokenizerFast.from_pretrained("neuralmind/bert-base-portuguese-cased")
        # English tokenizer
        tokenizer_en = BertTokenizerFast.from_pretrained("bert-base-uncased")

        return tokenizer_pt, tokenizer_en
