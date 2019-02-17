import random
import tensorflow as tf


def evaluate_list(preds, data_type=None, writer=None, global_step=0, subsample_ratio=None):
    avg_recall, avg_precision, avg_f1 = 0., 0., 0.

    if subsample_ratio is not None:
        preds = [preds[i] for i in random.sample(range(len(preds)), int(len(preds) * subsample_ratio))]

    for rec, prec, f_score in preds:
        avg_recall += rec
        avg_precision += prec
        avg_f1 += f_score

    avg_recall /= len(preds)
    avg_precision /= len(preds)
    avg_f1 /= len(preds)

    results = {
        'recall': avg_recall,
        'precision': avg_precision,
        'f1': avg_f1,
    }

    add_metric_summaries(results, data_type, writer, global_step)

    return results


def recall(y_true, y_pred, epsilon=1e-07):
    """ Recall metric. Computes a batch-wise average of recall for multi-label classification. """
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    possible_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)))

    rec = true_positives / (possible_positives + tf.constant(epsilon, dtype=tf.float32))
    return rec


def precision(y_true, y_pred, epsilon=1e-07):
    """ Precision metric. Computes a batch-wise average of precision for multi-label classification. """
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    predicted_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_pred, 0, 1)))

    prec = true_positives / (predicted_positives + tf.constant(epsilon, dtype=tf.float32))
    return prec


def harmonic_mean(x_1, x_2):
    """ Calculates the harmonic mean of two values """
    return 2 * ((x_1 * x_2) / (x_1 + x_2))


def f1(y_true, y_pred):
    """ Calculates the batch-wise F1 score for multi-label classification. """
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return harmonic_mean(prec, rec)


def add_metric_summaries(metrics, data_type=None, writer=None, global_step=0):
    """ Adds summaries for various metric functions.
        Args:
            metrics: dict of metric_name: value.
            data_type: String for whether we are in train/val.
            writer: Summary Writer object.
            global_step: Current step.
    """
    if writer is not None and data_type is not None:
        for key, value in metrics.items():
            summ = tf.Summary(value=[tf.Summary.Value(tag='{}/{}'.format(data_type, key), simple_value=value)])
            writer.add_summary(summ, global_step)
