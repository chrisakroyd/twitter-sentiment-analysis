import tensorflow as tf
from src import layers


class Attention(tf.keras.layers.Layer):
    def __init__(self, return_attention=False, epsilon=1e-07,
                 **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.supports_masking = True
        self.return_attention = return_attention
        self.epsilon = epsilon

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.kernel = self.add_weight(shape=(int(input_shape[-1]), 1, ),
                                      name='{}_W'.format(self.name),
                                      trainable=True)
        self.built = True

    def compute_mask(self, x, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None

    def call(self, x, training=None, mask=None):
        eij = tf.squeeze(tf.keras.backend.dot(x, self.kernel), axis=-1)
        eij = tf.tanh(eij)

        if mask is not None:
            eij = layers.apply_mask(eij, mask)

        a = tf.exp(eij)

        a /= tf.cast(tf.reduce_sum(a, axis=1, keepdims=True) + tf.keras.backend.epsilon(), dtype=tf.float32)
        att_weights = a

        weighted_input = x * tf.expand_dims(a, axis=-1)
        result = tf.reduce_sum(weighted_input, axis=1)

        if self.return_attention:
            return [result, att_weights]

        return result

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[1])]
        return input_shape[0], input_shape[-1]
