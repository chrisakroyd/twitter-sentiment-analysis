import tensorflow as tf


class HiddenBlock(tf.keras.Model):
    def __init__(self, units, use_bias=True, activation=None, dropout=0.1, skip_connection=False, **kwargs):
        """ Creates a hidden (dense) layer with the given parameters. This function facilitates creating variations of
            architectures based on command line / automated input.

            Block Structure:
               IN -> Hidden -> Skip Connection (Opt) -> Dropout -> OUT

            Args:
                units: Number of units in this layer.
                use_bias: Whether or not to add a bias term.
                activation: Activation function applied to this layer.
                dropout: The fraction of units to be dropped.
                skip_connection: Whether or not we utilise a residual skip connection.
            Returns:
                An initialized Keras RNN layer.
        """
        super(HiddenBlock, self).__init__(**kwargs)
        self.hidden = tf.keras.layers.Dense(units, use_bias=use_bias, activation=activation)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.skip_connection = skip_connection

    def call(self, x, training=None, mask=None):
        output = self.hidden(x)

        if self.skip_connection:
            output = output + x

        output = self.dropout(output, training=training)
        return output


class HiddenStack(tf.keras.Model):
    def __init__(self, units, layers, use_bias=True, activation=None, dropout=0.1, skip_connection=False, **kwargs):
        """ Builds a stack of hidden blocks.
            Args:
                units: Number of units in this layer.
                use_bias: Whether or not to add a bias term.
                layers: Number of hidden layers.
                activation: Activation function applied to this layer.
                dropout: The fraction of units to be dropped.
                skip_connection: Whether or not we utilise a residual skip connection.
            Returns:
                An initialized Keras layer that wraps a stack of hidden -> dropout blocks.
        """
        super(HiddenStack, self).__init__(**kwargs)
        self.hidden_layers = [HiddenBlock(units, dropout=dropout, use_bias=use_bias, activation=activation,
                                          skip_connection=skip_connection) for _ in range(layers)]

    def call(self, x, training=None, mask=None):
        for layer in self.hidden_layers:
            x = layer(x, training=training)
        return x
