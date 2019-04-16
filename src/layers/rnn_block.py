import tensorflow as tf
from src import constants


class RNNBlock(tf.keras.Model):
    def __init__(self, rnn_type, units, return_sequences=True, return_state=False, cudnn=True, bidirectional=True,
                 dropout=0.1, skip_connection=False, **kwargs):
        """ Creates an RNN of a given type with the given parameters. This function facilitates creating variations of
            architectures based on command line / automated input.

            Block Structure:
               IN -> RNN -> Skip Connection (Opt) -> Dropout -> OUT

            Args:
                rnn_type: Type of RNN, valid options are 'lstm' and 'gru'
                units: Number of units in this layer.
                return_sequences: See return_sequences parameter of keras LSTM/GRU.
                return_state: See return_state parameter of keras LSTM/GRU.
                cudnn: Whether or not to use the CuDNN implementation of the given rnn_type.
                bidirectional: Whether or not this is a bidirectional RNN and should be wrapped as such.
                dropout: The fraction of units to be dropped.
            Returns:
                An initialized Keras RNN layer.
        """
        super(RNNBlock, self).__init__(**kwargs)

        if rnn_type == constants.RNNTypes.GRU:
            if cudnn:
                self.rnn = tf.keras.layers.CuDNNGRU
            else:
                self.rnn = tf.keras.layers.GRU
        elif rnn_type == constants.RNNTypes.LSTM:
            if cudnn:
                self.rnn = tf.keras.layers.CuDNNLSTM
            else:
                self.rnn = tf.keras.layers.LSTM
        else:
            raise ValueError(constants.ErrorMessages.INVALID_RNN_TYPE.format(rnn_type=rnn_type))

        self.rnn = self.rnn(units, return_sequences=return_sequences, return_state=return_state)

        if bidirectional:
            self.rnn = tf.keras.layers.Bidirectional(self.rnn)

        self.dropout = tf.keras.layers.Dropout(dropout)
        self.skip_connection = skip_connection

    def call(self, x, training=None, mask=None):
        output = self.rnn(x)

        if self.skip_connection:
            output = output + x

        output = self.dropout(output, training=training)
        return output


class RNNStack(tf.keras.layers.Layer):
    def __init__(self, rnn_type, units, num_layers, return_sequences=True, cudnn=True, bidirectional=True,
                 dropout=0.1, skip_connection=False, **kwargs):
        """ Builds a stack of RNN blocks.

            TODO: Should be a tf.keras.Model, but throws no property error when it is, 0 impact but revist later.

            Args:
                rnn_type: Type of RNN, valid options are 'lstm' and 'gru'
                units: Number of units in this layer.
                num_layers: Number of RNN layers.
                return_sequences: See return_sequences parameter of keras LSTM/GRU.
                return_state: See return_state parameter of keras LSTM/GRU.
                cudnn: Whether or not to use the CuDNN implementation of the given rnn_type.
                bidirectional: Whether or not this is a bidirectional RNN and should be wrapped as such.
                dropout: The fraction of units to be dropped.
            Returns:
                An initialized Keras RNN layer.
        """
        super(RNNStack, self).__init__(**kwargs)
        self.rnn_layers = [RNNBlock(rnn_type, units, dropout=dropout,
                                    skip_connection=skip_connection, cudnn=cudnn,
                                    bidirectional=bidirectional, return_sequences=return_sequences)
                           for _ in range(num_layers)]

    def call(self, x, training=None, mask=None):
        for rnn in self.rnn_layers:
            x = rnn(x, training=training)
        return x
