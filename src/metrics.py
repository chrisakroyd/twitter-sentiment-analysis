import random
import tensorflow as tf
import numpy as np
import sklearn.metrics as metrics


def evaluate_list(preds, data_type=None, writer=None, global_step=0, subsample_ratio=None):
    recall, precision, f1 = 0., 0., 0.

    if subsample_ratio is not None:
        preds = [preds[i] for i in random.sample(range(len(preds)), int(len(preds) * subsample_ratio))]

    for _, pred, labels in preds:
        pred = np.argmax(pred.tolist(), axis=1)
        labels = labels.tolist()
        recall += metrics.recall_score(labels, pred, average='micro')
        precision += metrics.precision_score(labels, pred, average='micro')

    recall /= len(preds)
    precision /= len(preds)
    f1 = 2 * (precision * recall) / (precision + recall)

    results = {
        'recall': recall,
        'precision': precision,
        'f1': f1,
    }

    add_metric_summaries(results, data_type, writer, global_step)

    return results


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
