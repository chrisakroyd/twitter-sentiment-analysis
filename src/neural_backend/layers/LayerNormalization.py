from keras import backend as K
from keras import constraints, initializers, regularizers
from keras.engine.topology import Layer


class LayerNormalization(Layer):
    def __init__(self,
                 gamma_initializer='glorot_uniform',
                 beta_initializer='glorot_uniform',
                 gamma_regularizer=None,
                 beta_regularizer=None,
                 gamma_constraint=None,
                 beta_constraint=None,
                 eps=1e-6,
                 **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_initializer = initializers.get(beta_initializer)

        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_regularizer = regularizers.get(beta_regularizer)

        self.gamma_constraint = constraints.get(gamma_constraint)
        self.beta_constraint = constraints.get(beta_constraint)

        self.eps = eps

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=input_shape[-1:],
                                     initializer=self.gamma_initializer,
                                     name='gamma',
                                     regularizer=self.gamma_regularizer,
                                     constraint=self.gamma_constraint,
                                     trainable=True)

        self.beta = self.add_weight(shape=input_shape[-1:],
                                    initializer=self.beta_initializer,
                                    name='beta',
                                    regularizer=self.beta_regularizer,
                                    constraint=self.beta_constraint,
                                    trainable=True)
        super().build(input_shape)

    def call(self, x, mask=None):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape
