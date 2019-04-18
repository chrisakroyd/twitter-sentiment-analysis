import tensorflow as tf
import random


class RecordWriter(object):
    def __init__(self, max_tokens):
        """ RecordWriter class implements base functionality and utility methods for writing .tfrecord files
            Args:
                max_tokens: Maximum number of tokens per row, rows over this will be skipped by default.
        """
        self.max_tokens = max_tokens

    def float_list(self, values):
        return tf.train.Feature(float_list=tf.train.FloatList(value=values))

    def int_list(self, values):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

    def byte_list(self, values):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

    def create_feature_dict(self, data):
        """ Extracts features and encodes them as tf.train Feature's. """
        encoded_orig_tokens = [m.encode('utf-8') for m in data['orig_tokens']]
        encoded_tokens = [m.encode('utf-8') for m in data['tokens']]
        encoded_tags = [m.encode('utf-8') for m in data['tags']]

        features = {
            'orig_tokens': self.byte_list(encoded_orig_tokens),
            'tokens': self.byte_list(encoded_tokens),
            'tags': self.byte_list(encoded_tags),
            'num_tokens': self.int_list([data['num_tokens']]),
            'label': self.int_list([data['label']]),
        }

        return features

    def create_record(self, data):
        """ Creates a formatted tf.train Example for writing in a .tfrecord file. """
        features = self.create_feature_dict(data)
        record = tf.train.Example(features=tf.train.Features(feature=features))
        return record

    def shuffle(self, data):
        """ Takes a dict input and returns a shuffled list of its values. """
        shuffled = list(data)
        random.shuffle(shuffled)
        return shuffled

    def write(self, path, tweets, skip_too_long=True):
        """ Writes out context + queries for a dataset as a .tfrecord file, optionally skipping rows with too many
            tokens.

            Args:
                path: Filepath to write out a .tfrecord file
                tweets: Pre-processed tweets.
                skip_too_long: Boolean flag for whether rows > max_tokens are skipped or included.
        """
        shuffled_data = self.shuffle(tweets)

        with tf.python_io.TFRecordWriter(path) as writer:
            for data in shuffled_data:
                num_tokens = data['num_tokens']

                if num_tokens > self.max_tokens and skip_too_long:
                    continue

                record = self.create_record(data)
                writer.write(record.SerializeToString())
