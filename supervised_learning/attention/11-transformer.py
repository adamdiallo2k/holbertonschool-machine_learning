#!/usr/bin/env python3
"""
0-main.py

Demonstration code that:
 - imports the Transformer class from 0-transformer.py
 - creates an instance of Transformer
 - prints the types of encoder, decoder, and linear
   to match the required output
"""

import tensorflow as tf
Transformer = __import__('0-transformer').Transformer

if __name__ == "__main__":
    # Instantiate the Transformer network with the specified parameters
    transformer = Transformer(
        N=6,
        dm=512,
        h=8,
        hidden=2048,
        input_vocab=10000,
        target_vocab=12000,   # ensures linear has 12000 units
        max_seq_input=1000,
        max_seq_target=1500
    )

    # Print the desired outputs
    # 1) The type of the encoder (from 9-transformer_encoder.py)
    # 2) The type of the decoder (from 10-transformer_decoder.py)
    # 3) The type of the linear layer and its 'units' attribute
    print(type(transformer.encoder))
    print(type(transformer.decoder))
    print(type(transformer.linear), transformer.linear.units)
