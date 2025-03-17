#!/usr/bin/env python3
"""
Defines the Transformer network
"""
import tensorflow as tf

# Import the required Encoder and Decoder
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    """
    Creates a Transformer network
    """

    def __init__(self, N, dm, h, hidden, 
                 input_vocab, target_vocab,
                 max_seq_input, max_seq_target,
                 drop_rate=0.1):
        """
        Class constructor

        Args:
            N: number of blocks in the encoder and decoder
            dm: dimensionality of the model
            h: number of heads
            hidden: number of hidden units in the fully connected layers
            input_vocab: size of the input vocabulary
            target_vocab: size of the target vocabulary
            max_seq_input: maximum sequence length possible for the input
            max_seq_target: maximum sequence length possible for the target
            drop_rate: dropout rate

        Public instance attributes:
            encoder - the encoder layer
            decoder - the decoder layer
            linear - a final Dense layer with target_vocab units
        """
        super(Transformer, self).__init__()

        self.encoder = Encoder(N, dm, h, hidden,
                               input_vocab, max_seq_input,
                               drop_rate)
        self.decoder = Decoder(N, dm, h, hidden,
                               target_vocab, max_seq_target,
                               drop_rate)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training,
             encoder_mask, look_ahead_mask, decoder_mask):
        """
        Performs a forward pass through the Transformer.

        Args:
            inputs: tensor of shape (batch, input_seq_len) with the inputs
            target: tensor of shape (batch, target_seq_len) with the target
            training: boolean, whether the model is in training mode
            encoder_mask: the padding mask to apply to the encoder
            look_ahead_mask: the look-ahead mask to apply to the decoder
            decoder_mask: the padding mask to apply to the decoder

        Returns:
            A tensor of shape (batch, target_seq_len, target_vocab)
            containing the Transformer output
        """
        # Pass inputs through the encoder
        enc_output = self.encoder(inputs, training, encoder_mask)

        # Pass the target and encoder output through the decoder
        dec_output = self.decoder(target, enc_output, training,
                                  look_ahead_mask, decoder_mask)

        # Final linear projection
        final_output = self.linear(dec_output)  # (batch, target_seq_len, target_vocab)

        return final_output
