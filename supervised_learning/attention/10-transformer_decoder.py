#!/usr/bin/env python3
"""
Transformer Decoder (Refactored Commentary)
"""
import tensorflow as tf

# Import helper functions/classes from local modules
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """
    Defines the Transformer Decoder, which processes the output from
    the Encoder together with a target sequence to produce final
    decoded representations.
    """

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """
        Initializes the Transformer Decoder.

        Args:
            N (int): Number of decoder blocks.
            dm (int): Model dimensionality (size of embeddings).
            h (int): Number of attention heads.
            hidden (int): Dimensionality of the fully connected
                          'hidden' layer in each block.
            target_vocab (int): Size of the target vocabulary.
            max_seq_len (int): Maximum possible length of input sequences.
            drop_rate (float): Dropout rate to be used in the decoder.
        """
        super().__init__()
        self.N = N
        self.dm = dm

        # Embedding for the target tokens
        self.embedding = tf.keras.layers.Embedding(
            input_dim=target_vocab,
            output_dim=dm
        )

        # Precomputed positional encoding
        self.positional_encoding = positional_encoding(max_seq_len, dm)

        # Instantiate the N DecoderBlocks
        self.blocks = [
            DecoderBlock(dm, h, hidden, drop_rate) for _ in range(N)
        ]

        # Dropout layer after adding positional encoding
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Forward pass through the entire decoder stack.

        Args:
            x (tf.Tensor): Tokenized target input, shape (batch_size, target_seq_len).
            encoder_output (tf.Tensor): Encoder output, shape (batch_size, input_seq_len, dm).
            training (bool): Whether the model is in training mode.
            look_ahead_mask (tf.Tensor): Mask for the first multi-head attention
                                         (prevents attending to future tokens).
            padding_mask (tf.Tensor): Mask for the second multi-head attention
                                      (protects padded positions).

        Returns:
            tf.Tensor: The final decoder representation of shape (batch_size, target_seq_len, dm).
        """
        seq_len = x.shape[1]

        # Token embedding
        x = self.embedding(x)

        # Scale embeddings and add positional encodings
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len, :]

        # Apply dropout before processing blocks
        x = self.dropout(x, training=training)

        # Pass through each decoder block in turn
        for i in range(self.N):
            x = self.blocks[i](
                x, encoder_output, training,
                look_ahead_mask, padding_mask
            )

        return x
