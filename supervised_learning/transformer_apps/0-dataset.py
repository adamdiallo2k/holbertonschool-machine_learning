def tokenize_dataset(self, data):
    """
    Creates sub-word tokenizers for our dataset
    Args:
        data: tf.data.Dataset whose examples are formatted as a tuple (pt, en)
    Returns:
        tokenizer_pt: Portuguese tokenizer
        tokenizer_en: English tokenizer
    """
    # Extract the Portuguese and English sentences from the dataset
    pt_texts = []
    en_texts = []
    
    # Take all examples from the dataset for training the tokenizers
    for pt, en in data:
        pt_texts.append(pt.numpy().decode('utf-8'))
        en_texts.append(en.numpy().decode('utf-8'))

    # Initialize the Portuguese tokenizer with pre-trained model
    tokenizer_pt = BertTokenizerFast.from_pretrained(
        'neuralmind/bert-base-portuguese-cased')
    
    # Initialize the English tokenizer with pre-trained model
    tokenizer_en = BertTokenizerFast.from_pretrained(
        'bert-base-uncased')
    
    # Train the tokenizers with maximum vocabulary size of 2^13
    vocab_size = 2**13
    
    # For BERT tokenizers, we need to resize the vocabulary to match requirements
    tokenizer_pt.resize_token_embeddings(vocab_size)
    tokenizer_en.resize_token_embeddings(vocab_size)
    
    return tokenizer_pt, tokenizer_en
