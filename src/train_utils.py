import tensorflow as tf
import warnings


def ema_ops(train_op, decay, variables=None):
    """ Adds ops to calculate and maintain Exponential moving average over all trainable weights.
        Args:
            train_op: Result of optimizer.minimize or optimizer.apply_gradients
            decay: Exponential decay rate.
            variables: List of variables to maintain average over.
        Returns:
            Modified train_op and ema object.
    """
    if variables is None:
        variables = tf.trainable_variables()

    with tf.name_scope('ema_ops'):
        ema = tf.train.ExponentialMovingAverage(decay=decay)
        maintain_avg_op = ema.apply(variables)
        with tf.control_dependencies([train_op]):
            train_op = tf.group(maintain_avg_op)
            return train_op, ema


def l2_ops(l2, variables=None):
    """ Adds ops to calculate l2 loss over all trainable variables.
        Args:
            l2: L2 penalty to apply.
            variables: A list of trainable variables, if None, uses tf.trainable_variables()
        Returns:
            Tensor for the sum of l2_loss over all given variables.
    """
    if variables is None:
        variables = tf.trainable_variables()

    with tf.name_scope('l2_ops'):
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in variables]) * l2
        return l2_loss


def clip_by_global_norm(gradients, clip_norm=5.0):
    """ Applies clipping by global norm to a list of gradient tensors.
        Args:
            gradients: list of gradient tensors.
            clip_norm: Norm of the gradient value to clip by.
        Returns:
            Zipped gradients and variables.
    """
    gradients, variables = zip(*gradients)
    clipped_grads, _ = tf.clip_by_global_norm(gradients, clip_norm)
    grads_and_vars = zip(clipped_grads, variables)
    return grads_and_vars


def inverse_exponential_warmup(learn_rate, warmup_steps=1000, global_step=None):
    """ Learning rate warmup with an inverse exponential curve.
        Args:
            learn_rate: Learn rate.
            warmup_steps: Number of steps till we reach the given learn_rate.
            global_step: Global step variable.
        Returns:
            An op to calculate the learn rate at a given step.
    """
    if global_step is None:
        global_step = tf.train.get_global_step()
    else:
        tf.train.assert_global_step(global_step)

    lr_decay = learn_rate / tf.log(tf.to_float(warmup_steps)) * tf.log(tf.to_float(global_step) + 1)
    return tf.minimum(learn_rate, lr_decay)


def linear_warmup(learn_rate, warmup_steps=1000, global_step=None):
    """ Learning rate warmup at a linear rate.
        Args:
            learn_rate: Learn rate.
            warmup_steps: Number of steps till we reach the given learn_rate.
            global_step: Global step variable.
        Returns:
            An op to calculate the learn rate at a given step.
    """
    if global_step is None:
        global_step = tf.train.get_global_step()
    else:
        tf.train.assert_global_step(global_step)

    lr_decay = tf.minimum(1.0, tf.to_float(global_step) / warmup_steps)
    return lr_decay * learn_rate


def get_warmup_scheme(name, learn_rate, warmup_steps=1000, global_step=None):
    """ Gets and returns the named warmup scheme.
        Args:
            name: Name of the warmup scheme, either linear or inverse_exp otherwise simply returns learn rate.
            learn_rate: Learn rate.
            warmup_steps: Number of steps till we reach the given learn_rate.
            global_step: Global step variable.
        Returns:
            An op to calculate the learn rate at a given step.
    """
    name = name.lower().strip()
    if name == 'linear':
        return linear_warmup(learn_rate, warmup_steps, global_step)
    elif name == 'inverse_exp':
        return inverse_exponential_warmup(learn_rate, warmup_steps, global_step)
    else:
        warnings.warn('Invalid warmup scheme specified, using learn rate.')
        return learn_rate


def construct_train_op(loss, learn_rate=0.001, warmup_scheme='inverse_exp', warmup_steps=1000, clip_norm=5.0,
                       ema_decay=0.999, beta1=0.8, beta2=0.999, epsilon=1e-7, global_step=None):
    """ Constructs a training op with options for warmup schemes, gradient clipping and
        exponential moving average weight decay.

        Args:
            loss: A tensor full of loss values.
            learn_rate: Learn rate.
            warmup_scheme: Name of the warmup scheme to utilise, if None or invalid just uses learn_rate.
            warmup_steps: Number of steps till we reach the given learn_rate.
            clip_norm: A scalar float.
            ema_decay: Decay rate to use.
            beta1: beta1 param for the adam optimizer.
            beta2: beta2 param for the adam optimizer.
            epsilon: A small constant for numerical stability.
            global_step: Global step variable.
        Returns:

    """
    if global_step is None:
        global_step = tf.train.get_global_step()
    else:
        tf.train.assert_global_step(global_step)

    learning_rate = get_warmup_scheme(warmup_scheme, learn_rate, warmup_steps, global_step)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       beta1=beta1,
                                       beta2=beta2,
                                       epsilon=epsilon)

    if clip_norm > 0.0:
        grads = optimizer.compute_gradients(loss)
        grads_and_vars = clip_by_global_norm(grads, clip_norm)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    else:
        train_op = optimizer.minimize(loss, global_step=global_step)

    if 0.0 < ema_decay < 1.0:
        train_op, _ = ema_ops(train_op, ema_decay)

    return train_op


def inputs_as_tuple(placeholders):
    """ Converts the placeholder dict generated by our pipeline to a tuple of the network inputs.
        Args:
            placeholders: A dict of input tensors.
        Returns:
            A tuple of input tensors.
    """
    keys = ['context_words', 'context_chars', 'context_length', 'query_words', 'query_chars', 'query_length']
    return dict_keys_as_tuple(placeholders, keys)


def labels_as_tuple(placeholders):
    """ Converts the placeholder dict generated by our pipeline to a tuple of the network labels.
        Args:
            placeholders: A dict of input tensors.
        Returns:
            A tuple of input tensors.
    """
    keys = ['answer_starts', 'answer_ends', 'answer_id']
    return dict_keys_as_tuple(placeholders, keys)


def dict_keys_as_tuple(placeholders, keys=None):
    """
        Args:
            placeholders: A dict of input tensors.
            keys: List of string keys on the given placeholder dict.
        Returns:
            A tuple of input tensors.
    """
    if keys is None:
        raise ValueError('No keys given to dict_keys_as_tuple.')
    return tuple([placeholders[key] for key in keys])


def get_saver(ema_decay=0.0, ema_vars_only=False):
    """
        Args:
            ema_decay: Ema decay value.
            ema_vars_only: Boolean flag for restoring EMA variables only.
        Returns:
            A tuple of input tensors.
    """
    if 0.0 < ema_decay < 1.0 and ema_vars_only:
        variable_averages = tf.train.ExponentialMovingAverage(0.)
        return tf.train.Saver(variable_averages.variables_to_restore())
    return tf.train.Saver()
