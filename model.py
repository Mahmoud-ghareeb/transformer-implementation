import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers as tfl

class InputEmbedding(tfl.Layer):
    def __init__(self, vocab_size: int, d_model: int):
        super(InputEmbedding, self).__init__()

        self.d_model = d_model
        self.embedding = tfl.Embedding(vocab_size, d_model)

    def call(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

# x = tf.random.uniform(shape=[1, 5], minval=1, maxval=1000, dtype=tf.int32)
# print(InputEmbedding(10000, 512)(x))

class PositionalEncoding(tfl.Layer):
    def __init__(self, d_model: int, seq_length: int):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(seq_length, d_model)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, seq_length, d_model):
        angle_rads = self.get_angles(np.arange(seq_length)[:, np.newaxis], # => (seq_len, 1)
                                     np.arange(d_model)[np.newaxis, :], # => (1, d_model)
                                     d_model)

        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]
    
# x = tf.random.normal([1, 6, 512])
# print(PositionalEncoding(512, 6)(x))

class LayerNormalization(tfl.Layer):
    def __init__(self, eps: float = 1e-7):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.alfa = tf.Variable(tf.ones(1), trainable=True)
        self.beta = tf.Variable(tf.ones(1), trainable=True)
    def call(self, x):
        mean = tf.math.reduce_mean(x, axis=-1, keepdims=True)
        std = tf.math.reduce_std(x, axis=-1, keepdims=True)

        return self.alfa * (x - mean) / (std + self.eps) + self.beta

    
class FeedForwardLayer(tfl.Layer):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super(FeedForwardLayer, self).__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.dense1 = tfl.Dense(d_ff)
        self.activation = tfl.Activation('relu')
        self.dropout = tfl.Dropout(dropout)
        self.dense2 = tfl.Dense(d_model)

    def call(self, x):
        return self.dense2(self.dropout(self.activation(self.dense1(x))))
    
class MultiHeadAttention(tfl.Layer):
    def __init__(self, d_model: int, h: int, dropout: float = .1):
        super(MultiHeadAttention, self).__init__()

        self.h = h
        self.d_model = d_model
        self.dropout = dropout
        self.d_k = d_model // h
        self.w_q = tfl.Dense(d_model)
        self.w_k = tfl.Dense(d_model)
        self.w_v = tfl.Dense(d_model)
        self.w_o = tfl.Dense(d_model)

    def call(self, q, k, v, mask = None):
        query = self.w_q(q) # => (batch, seq, d_model)
        key = self.w_k(k) # => (batch, seq, d_model)
        value = self.w_v(v) # => (batch, seq, d_model)

        # => (batch, seq, h, d_k) => (batch, h, seq, d_k)
        query = tf.transpose(tf.reshape(query, [query.shape[0], query.shape[1], self.h, self.d_k]), [0, 2, 1, 3])
        # => (batch, seq, h, d_k) => (batch, h, seq, d_k)
        key = tf.transpose(tf.reshape(key, [key.shape[0], key.shape[1], self.h, self.d_k]), [0, 2, 1, 3])
        # => (batch, seq, h, d_k) => (batch, h, seq, d_k)
        value = tf.transpose(tf.reshape(value, [value.shape[0], value.shape[1], self.h, self.d_k]), [0, 2, 1, 3])

        # => (batch, h, seq, seq)
        attention_scores = (query @ tf.transpose(key, [0, 1, 3, 2])) / math.sqrt(self.d_k)
        if mask is not None:
            attention_scores = tf.where(mask == 0, -1e9, attention_scores)
        attention_scores = tf.nn.softmax(attention_scores, axis=-1)

        outputs = attention_scores @ value
        outputs = tf.transpose(outputs, [0, 2, 1, 3]) # => (batch, seq, h, d_k)
        outputs = tf.reshape(outputs, [outputs.shape[0], outputs.shape[1], outputs.shape[2] * outputs.shape[3]]) # => (batch, seq, d_model)

        return self.w_o(outputs)

class AddAndNorm(tfl.Layer):
    def __init__(self, dropout: float):
        super(AddAndNorm, self).__init__()

        self.dropout = tfl.Dropout(dropout)
        self.norm = LayerNormalization()

    def call(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderBlock(tfl.Layer):
    def __init__(self,
                 d_model: int, 
                 h: int, 
                 dropout: float,
                 d_ff: int):
        super(EncoderBlock, self).__init__()

        self.mha = MultiHeadAttention(d_model, h, dropout)
        self.aan1 = AddAndNorm(dropout)
        self.ffl = FeedForwardLayer(d_model, d_ff, dropout)
        self.aan2 = AddAndNorm(dropout)

    def call(self, x):
        x = self.aan1(x, lambda x: self.mha(x, x, x))
        x = self.aan2(x, lambda x: self.ffl(x))

        return x

class Encoder(Model):
    def __init__(self,
                 vocab_size: int,
                 seq_length: int,
                 d_model: int,
                 h: int,
                 dropout: float,
                 d_ff: int,
                 n: int):
        super(Encoder, self).__init__()

        self.embedding = InputEmbedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, seq_length)
        self.encs = [EncoderBlock(d_model, h, dropout, d_ff) for _ in range(n)]

    def call(self, x):
        x = self.embedding(x)
        x = x + self.pos(x)
        for enc in self.encs:
            x = enc(x)

        return x

# x = tf.random.uniform(shape=[1, 5], minval=1, maxval=1000, dtype=tf.int32)
# print(Encoder(10000, 6, 512, 8, .1, 2048, 6)(x))

class DecoderBlock(tfl.Layer):
    def __init__(self,
                 d_model: int,
                 h: int,
                 dropout: float,
                 d_ff: int):
        super(DecoderBlock, self).__init__()

        self.masked_mha = MultiHeadAttention(d_model, h, dropout)
        self.aan1 = AddAndNorm(dropout)
        self.cross_mha = MultiHeadAttention(d_model, h, dropout)
        self.aan2 = AddAndNorm(dropout)
        self.ffl = FeedForwardLayer(d_model, d_ff, dropout)
        self.aan3 = AddAndNorm(dropout)

    def call(self, x, encoder_output, enc_mask, dec_mask):
        x = self.aan1(x, lambda x: self.masked_mha(x, x, x, dec_mask))
        x = self.aan2(x, lambda x: self.cross_mha(x, encoder_output, encoder_output, enc_mask))
        x = self.aan3(x, lambda x: self.ffl(x))

        return x

class Decoder(Model):
    def __init__(self,
                 vocab_size: int,
                 seq_length: int,
                 d_model: int,
                 h: int,
                 dropout: float,
                 d_ff: int,
                 n: int):
        super(Decoder, self).__init__()

        self.embedding = InputEmbedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, seq_length)
        self.decs = [DecoderBlock(d_model, h, dropout, d_ff) for _ in range(n)]

    def call(self, x, encoder_output):
        x = self.embedding(x)
        x = x + self.pos(x)
        enc_mask = True
        dec_mask = True
        for dec in self.decs:
            x = dec(x, encoder_output, enc_mask, dec_mask)

        return x

# y = tf.random.uniform(shape=[1, 5], minval=1, maxval=1000, dtype=tf.int32)
# print(Decoder(10000, 6, 512, 8, .1, 2048, 6)(y))

class ProjectionLayer(tfl.Layer):
    def __init__(self, vocab_size: int):
        super(ProjectionLayer, self).__init__()

        self.dense = tfl.Dense(vocab_size)

    def call(self, x):
        return self.dense(x)
    
class Transformer(Model):
    def __init__(self,
                 enc_vocab_size: int,
                 enc_seq_length: int,
                 dec_vocab_size: int,
                 dec_seq_length: int,
                 d_model: int = 512,
                 h: int = 8,
                 dropout: float = .3,
                 d_ff: int = 2048,
                 n: int = 6):
        super(Transformer, self).__init__()

        self.encoder = Encoder(enc_vocab_size, enc_seq_length, d_model, h, dropout, d_ff, n)
        self.decoder = Decoder(dec_vocab_size, dec_seq_length, d_model, h, dropout, d_ff, n)
        self.dense = ProjectionLayer(enc_vocab_size)

    def call(self, enc_inputs, dec_inputs):
        encoder_outputs = self.encoder(enc_inputs)
        decoder_outputs = self.decoder(dec_inputs, encoder_outputs)
        logits = self.dense(decoder_outputs)

        return logits



if __name__ == '__main__':

    x = tf.random.uniform(shape=[1, 5], minval=1, maxval=1000, dtype=tf.int32)
    y = tf.random.uniform(shape=[1, 7], minval=5, maxval=5000, dtype=tf.int32)
    print(Transformer(10000, 5, 10000, 7)(x, y))