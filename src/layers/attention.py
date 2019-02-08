import tensorflow as tf


class Attention(tf.keras.layers.Layer):
    def __init__(self,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 return_attention=False,
                 **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.supports_masking = True
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.return_attention = return_attention

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.kernel = self.add_weight(shape=(int(input_shape[-1]), 1, ),
                                      initializer=self.kernel_initializer,
                                      name='{}_W'.format(self.name),
                                      regularizer=self.kernel_regularizer,
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

        # apply mask.
        if mask is not None:
            eij *= tf.cast(mask, tf.float32)

        a = tf.exp(eij)

        a /= tf.cast(tf.reduce_sum(a, axis=1, keepdims=True) + tf.keras.backend.epsilon(), tf.float32)
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
