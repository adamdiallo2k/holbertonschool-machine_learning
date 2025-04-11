#!/usr/bin/env python3
"""Bag of Words module for creating text embeddings"""
import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix
    
    Args:
        sentences: list of sentences to analyze
        vocab: list of the vocabulary words to use for the analysis
               If None, all words within sentences should be used
    
    Returns:
        embeddings: numpy.ndarray of shape (s, f) containing the embeddings
                   where s is the number of sentences and f is the number of features
        features: list of the features used for embeddings
    """
    # Process sentences to extract words
    processed_sentences = []
    all_words = set()
    
    for sentence in sentences:
        # Convert to lowercase and remove punctuation
        words = re.findall(r'\b[a-zA-Z]+\b', sentence.lower())
        processed_sentences.append(words)
        all_words.update(words)
    
    # Create vocabulary
    if vocab is None:
        features = sorted(list(all_words))
    else:
        features = vocab
    
    # Create word to index mapping
    word_to_idx = {word: idx for idx, word in enumerate(features)}
    
    # Create embeddings matrix
    num_sentences = len(sentences)
    num_features = len(features)
    embeddings = np.zeros((num_sentences, num_features), dtype=int)
    
    # Fill embeddings matrix
    for i, words in enumerate(processed_sentences):
        for word in words:
            if word in word_to_idx:
                embeddings[i, word_to_idx[word]] += 1
    
    # Convert features to numpy array to match expected output format
    features = np.array(features)
    
    return embeddings, features
