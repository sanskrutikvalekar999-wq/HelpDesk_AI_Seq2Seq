import tensorflow as tf
from tensorflow.keras import layers


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units):
        super().__init__()
        self.embedding = layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.lstm = layers.LSTM(
            units,
            return_sequences=True,
            return_state=True,
        )

    def call(self, x, training=False):
        x = self.embedding(x)
        outputs, h, c = self.lstm(x, training=training)
        return outputs, h, c


class BahdanauAttention(layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, query, values, value_mask=None):
        # query: (batch, hidden)
        # values: (batch, time, hidden)
        query_time = tf.expand_dims(query, 1)
        score = self.V(
            tf.nn.tanh(self.W1(values) + self.W2(query_time))
        )

        if value_mask is not None:
            mask = tf.cast(tf.expand_dims(value_mask, -1), score.dtype)
            score = tf.where(mask > 0, score, tf.fill(tf.shape(score), -1e9))

        attention_weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(attention_weights * values, axis=1)
        return context, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units, attention):
        super().__init__()
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.lstm = layers.LSTM(
            units,
            return_sequences=True,
            return_state=True,
        )
        self.fc = layers.Dense(vocab_size)
        self.attention = attention

    def call(self, x, hidden, cell, encoder_outputs, encoder_mask=None, training=False):
        context, attention_weights = self.attention(
            hidden, encoder_outputs, encoder_mask
        )

        x = self.embedding(x)
        context = tf.expand_dims(context, 1)
        context = tf.repeat(context, tf.shape(x)[1], axis=1)
        x = tf.concat([context, x], axis=-1)

        output, h, c = self.lstm(
            x, initial_state=[hidden, cell], training=training
        )
        logits = self.fc(output)
        return logits, h, c, attention_weights
