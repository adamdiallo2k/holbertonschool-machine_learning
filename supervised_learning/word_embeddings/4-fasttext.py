#!/usr/bin/env python3
"""Defines a function that trains a Gensim FastText model."""

from gensim.models import FastText

def fasttext_model(sentences, vector_size=100, min_count=5, negative=5, window=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Creates and trains a Gensim FastText model.

    Args:
        sentences (list of list of str): The sentences to be trained on
        vector_size (int): Dimensionality of the embedding layer
        min_count (int): Minimum frequency for a word to be included in training
        negative (int): Size of negative sampling
        window (int): Maximum distance between the current and predicted word
        cbow (bool): Training type; True uses CBOW, False uses Skip-gram
        epochs (int): Number of training epochs
        seed (int): Random seed for reproducibility
        workers (int): Number of worker threads to use

    Returns:
        FastText: The trained FastText model
    """
    sg = 0 if cbow else 1

    # Initialize the FastText model
    model = FastText(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        negative=negative,
        sg=sg,
        seed=seed,
        workers=workers
    )

    # Build the vocabulary
    model.build_vocab(sentences)

    # Train the FastText model
    model.train(
        sentences=sentences,
        total_examples=len(sentences),
        epochs=epochs
    )

    return model
