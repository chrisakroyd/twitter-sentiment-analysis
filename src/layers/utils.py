import tensorflow as tf


def create_mask(lengths, maxlen):
    """ Create a mask of the given length.
        Args:
            lengths: Length of each item [batch_size]
            maxlen: A scalar value for the maximum length within the batch.
        Returns:
            A boolean tensor of shape [batch_size, maxlen]
    """
    return tf.sequence_mask(lengths, maxlen=maxlen)


def apply_mask(inputs, mask, mask_value=-1e12):
    """ Exponential mask for logits.
        During exponent operation 0 becomes 1, therefore instead of multiplying directly by the boolean mask,
        we create a mask which fills False positions with a large negative value. Note: Should always be applied
        before softmax.

         Args:
            inputs: Arbitrary-rank logits tensor to be masked.
            mask: A tf.bool mask tensor.
            mask_value: A scalar value to fill `False` positions.
        Returns:
            Masked inputs with the same shape as `inputs`.
    """
    return inputs + create_mask_vector(mask, mask_value)


def create_mask_vector(mask, mask_value=-1e12):
    """ Converts a tf.bool tensor into a float32 mask filled with mask_value for padding positions. """
    return tf.cast(tf.logical_not(mask), dtype=tf.float32) * mask_value
