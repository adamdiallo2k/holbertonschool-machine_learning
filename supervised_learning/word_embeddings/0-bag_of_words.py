#!/usr/bin/env python3
"""Bag of Words module for creating text embeddings"""
import numpy as np


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
    # Clean and tokenize sentences
    cleaned_sentences = []
    vocab_set = set()
    
    for sentence in sentences:
        # Convert to lowercase
        sentence = sentence.lower()
        # Remove punctuation and split
        words = []
        curr_word = ""
        for char in sentence:
            if char.isalpha():
                curr_word += char
            elif curr_word:
                words.append(curr_word)
                curr_word = ""
        if curr_word:  # Add the last word if it exists
            words.append(curr_word)
        
        cleaned_sentences.append(words)
        vocab_set.update(words)
    
    # Create features list
    if vocab is None:
        features = np.array(sorted(list(vocab_set)))
    else:
        features = np.array(vocab)
    
    # Create embeddings matrix
    num_sentences = len(sentences)
    num_features = len(features)
    embeddings = np.zeros((num_sentences, num_features), dtype=int)
    
    # Word to index mapping
    word_to_idx = {word: idx for idx, word in enumerate(features)}
    
    # Populate embeddings
    for i, words in enumerate(cleaned_sentences):
        for word in words:
            if word in word_to_idx:
                embeddings[i, word_to_idx[word]] += 1
    
    return embeddings, features
