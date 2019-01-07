import tensorflow as tf


class Attention(tf.keras.layers.Layer):
    def __init__(self,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 use_bias=True,
                 return_attention=False,
                 **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.supports_masking = True
        # Weights/bias initializers e.g. glorut, zeros
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        # Regularization support.
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)

        self.use_bias = use_bias
        self.return_attention = return_attention

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.kernel = self.add_weight(shape=(int(input_shape[-1]), 1, ),
                                      initializer=self.kernel_initializer,
                                      name='{}_W'.format(self.name),
                                      regularizer=self.kernel_regularizer,
                                      trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(shape=(int(input_shape[1]), ),
                                        initializer=self.bias_initializer,
                                        name='{}_b'.format(self.name),
                                        regularizer=self.bias_regularizer,
                                        trainable=True)
        else:
            self.bias = None

        self.built = True

    def compute_mask(self, x, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None

    def call(self, x, training=None, mask=None):
        eij = tf.squeeze(tf.keras.backend.dot(x, self.kernel), axis=-1)

        if self.use_bias:
            eij += self.bias

        eij = tf.tanh(eij)

        a = tf.exp(eij)

        # apply mask.
        if mask is not None:
            a *= tf.cast(mask, tf.float32)

        a /= tf.cast(tf.reduce_sum(a, axis=1, keepdims=True) + tf.keras.backend.epsilon(), tf.float32)

        att_weights = a

        a = tf.expand_dims(a, axis=-1)
        # Calculate the weighted sum.
        weighted_input = x * a

        result = tf.reduce_sum(weighted_input, axis=1)

        if self.return_attention:
            return [result, att_weights]

        return result

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[1])]
        return input_shape[0], input_shape[-1]
