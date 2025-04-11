#!/usr/bin/env python3
import numpy as np
import re
import string

def bag_of_words(sentences, vocab=None):
    # If a vocab is provided, standardize it (i.e. lowercase, remove possessives and punctuation)
    if vocab is not None:
        features = []
        for word in vocab:
            # Lowercase the word
            word_clean = word.lower()
            # Remove possessive 's if it exists
            word_clean = re.sub(r"'s\b", "", word_clean)
            # Remove any remaining punctuation
            word_clean = word_clean.translate(str.maketrans("", "", string.punctuation))
            features.append(word_clean)
    else:
        # No vocab provided, so build set of unique words from all sentences.
        words_set = set()
        for sentence in sentences:
            # Lowercase and remove possessive 's and punctuation
            s = sentence.lower()
            s = re.sub(r"'s\b", "", s)
            s = s.translate(str.maketrans("", "", string.punctuation))
            words = s.split()
            words_set.update(words)
        features = sorted(list(words_set))
    
    # Create a mapping from word to index in the features list.
    word2idx = {word: idx for idx, word in enumerate(features)}
    
    # Build the embeddings matrix: one row per sentence and one column per feature.
    E = np.zeros((len(sentences), len(features)), dtype=int)
    
    # Loop through each sentence and count occurrences of each word.
    for i, sentence in enumerate(sentences):
        s = sentence.lower()
        s = re.sub(r"'s\b", "", s)
        s = s.translate(str.maketrans("", "", string.punctuation))
        words = s.split()
        for word in words:
            if word in word2idx:
                E[i, word2idx[word]] += 1
                
    return E, features
