#!/usr/bin/env python3
"""
    Module to create Class RNN Decoder
"""
import tensorflow as tf

SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
        class to create RNN decoder for machine translation
    """

    def __init__(self, vocab, embedding, units, batch):
        """
            class constructor

        :param vocab: integer, size of output vocabulary
        :param embedding: integer, dimensionality of embedding vector
        :param units: integer, number hidden units in RNN cell
        :param batch: integer, batch size
        """
        invalid_args = [arg for arg in [vocab, embedding, units, batch]
                        if not isinstance(arg, int)]
        if invalid_args:
            arg_str = ", ".join([f"{arg}" for arg in invalid_args])
            raise TypeError(f"{arg_str} Should be an integer.")

        super().__init__()
        self.units = units
        self.batch = batch
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        self.gru = tf.keras.layers.GRU(
            units=units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform")
        self.F = tf.keras.layers.Dense(units=vocab)
        self.attention = SelfAttention(self.units)

    def call(self, x, s_prev, hidden_states):
        """
            call function

        :param x: tensor, shape(batch,1), previous word in the target
        :param s_prev: tensor, shape(batch, units) previous decoder
            hidden state
        :param hidden_states: tensor, shape(batch,input_seq_len,units)
             outputs of the encoder

        :return: y, s
            y: tensor, shape(batch, vocab) output word as a one hot
                vector in the target vocabulary
            s: tensor, shape(batch, units) new decoder hidden state
        """
        # context and weigh : context.shape(32,256)
        context, att_weights = self.attention(s_prev, hidden_states)

        # embedding vector
        x = self.embedding(x)  # shape(32, 1, 128)

        # concatenate context with embedding vector
        context = tf.expand_dims(context, axis=1)  # context.shape(32,1,256)
        context_concat = tf.concat([context, x], axis=-1)
        # context.shape(32,1,384)

        outputs, hidden_state = self.gru(context_concat)
        # output.shape(32,1,256)

        # new_output.shape(32,256)
        new_outputs = tf.reshape(outputs,
                                 shape=(outputs.shape[0], outputs.shape[2]))

        y = self.F(new_outputs)

        return y, hidden_state
