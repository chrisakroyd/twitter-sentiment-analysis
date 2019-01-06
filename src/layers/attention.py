from keras import backend as K
from keras.engine.topology import Layer
from keras import constraints, initializers, regularizers


def dot_product(x, kernel):
    if K.backend() == 'tensorflow':
        # todo: check that this is correct
        return K.squeeze(K.dot(x, kernel), axis=-1)
    else:
        return K.dot(x, kernel)


class Attention(Layer):
    def __init__(self,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 use_bias=True,
                 return_attention=False,
                 **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.supports_masking = True
        # Weights/bias initializers e.g. glorut, zeros
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        # Regularization support.
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        # Any constraints
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.use_bias = use_bias
        self.return_attention = return_attention

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.kernel = self.add_weight(shape=(input_shape[-1], 1),
                                      initializer=self.kernel_initializer,
                                      name='{}_W'.format(self.name),
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(shape=(input_shape[1],),
                                        initializer=self.bias_initializer,
                                        name='{}_b'.format(self.name),
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None

    def call(self, x, mask=None):
        eij = dot_product(x, self.kernel)

        if self.use_bias:
            eij += self.bias

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask.
        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        att_weights = a

        a = K.expand_dims(a)
        # Calculate the weighted sum.
        weighted_input = x * a

        result = K.sum(weighted_input, axis=1)

        if self.return_attention:
            return [result, att_weights]

        return result

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[1])]
        return input_shape[0], input_shape[-1]