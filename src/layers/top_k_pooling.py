import tensorflow as tf


class TopKPooling(tf.keras.layers.Layer):
    def __init__(self, k=5, **kwargs):
        """ Top K Pooling implementation

            Similar to GlobalMaxPool, except instead of returning the Top-1 Max, returns the Top-k max values per input
            channel.

            Args:
                k: Number of max operations to perform.
        """
        super(TopKPooling, self).__init__(**kwargs)
        self.k = k

    def call(self, x, training=None, mask=None):
        """ Call function detailing this layers ops.
            Args:
                x: A single tensor of shape [batch_size, seq_length, ?]
                training: Boolean flag for training mode.
                mask: A mask tensor.
        """
        units = x.shape[2]
        x = tf.transpose(x, [0, 2, 1])
        k_max = tf.math.top_k(x, k=self.k)
        return tf.reshape(k_max[0], (-1, units * self.k))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2] * self.k)
