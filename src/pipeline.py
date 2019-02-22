import os
import tensorflow as tf
# useful link on pipelines: https://cs230-stanford.github.io/tensorflow-input-data.html


def tf_record_pipeline(filenames, buffer_size=1024, num_parallel_calls=4):
    """ Creates a dataset from a TFRecord file.
        Args:
            filenames: A list of paths to .tfrecord files.
            buffer_size: Number of records to buffer.
            num_parallel_calls: How many functions we run in parallel.
        Returns:
            A `tf.data.Dataset` object.
    """
    int_feature = tf.FixedLenFeature([], tf.int64)
    str_feature = tf.FixedLenSequenceFeature([], tf.string, allow_missing=True)

    features = {
        'tokens': str_feature,
        'tags': str_feature,
        'num_tokens': int_feature,
        'label': int_feature,
    }

    def parse(proto):
        return tf.parse_single_example(proto, features=features)

    data = tf.data.TFRecordDataset(filenames,
                                   buffer_size=buffer_size,
                                   num_parallel_reads=num_parallel_calls)

    data = data.map(parse, num_parallel_calls=num_parallel_calls)
    return data


def index_lookup(data, tables, char_limit=16, num_parallel_calls=4):
    """ Adds a map function to the dataset that maps strings to indices.

        To save memory + hard drive space we store contexts and queries as tokenised strings. Therefore we need
        to perform two tasks; Extract characters and map words + chars to an index for the embedding layer.

        @TODO This is a pretty ugly solution to support labelled/unlabelled modes, refactor target?

        Args:
            data: A `tf.data.Dataset` object.
            tables: A tuple of contrib.lookup tables mapping string words to indices and string characters to indices.
            char_limit: Max number of characters per word.
            num_parallel_calls: An int for how many parallel lookups we perform.
            has_labels: Include labels in the output dict.
        Returns:
            A `tf.data.Dataset` object.
    """
    word_table, char_table, tag_table = tables

    def _lookup(fields):
        # +1 allows us to use 0 as a padding character without explicitly mapping it.
        tokens = word_table.lookup(fields['tokens']) + 1
        # @TODO Revist the +1's -1 situation.
        # Get chars + lookup in table, as table is 0 indexed, we have the default at -1 for the pad which becomes 0
        # with the addition of 1 to again treat padding as 0 without needing to define a padding character.
        chars = tf.string_split(fields['tokens'], delimiter='')
        chars = tf.sparse.to_dense(char_table.lookup(chars), default_value=-1) + 1
        chars = chars[:, :char_limit]

        tags = tag_table.lookup(fields['tags'])

        fields['words'] = tokens
        fields['chars'] = chars
        fields['tags'] = tags

        return fields

    data = data.map(_lookup, num_parallel_calls=num_parallel_calls)
    return data


def format_output(data, num_classes, num_tags, num_parallel_calls=4):
    """ Casts tensors to their intended dtypes and converts label tensor to be one-hot. """

    def _lookup(fields):
        out_dict = {
            'words': tf.cast(fields['words'], dtype=tf.int32),
            'chars': tf.cast(fields['chars'], dtype=tf.int32),
            'tags': tf.one_hot(fields['tags'], num_tags, dtype=tf.float32),
            'num_tokens': tf.cast(fields['num_tokens'], dtype=tf.int32),
        }

        if 'label' in fields:
            out_dict.update({
                'label': tf.one_hot(fields['label'], num_classes, dtype=tf.int32)
            })

        return out_dict

    data = data.map(_lookup, num_parallel_calls=num_parallel_calls)

    return data


def create_buckets(bucket_size, max_size, bucket_ranges=None):
    """ Optionally generates bucket ranges if they aren't specified in the hparams.
        Args:
            bucket_size: Size of the bucket.
            max_size: Maximum length of the thing we want to bucket.
            bucket_ranges: Pre-generated bucket ranges.
        Returns:
            A list of integers for the start of buckets.
    """
    if bucket_ranges is None or len(bucket_ranges) == 0:
        return [i for i in range(0, max_size + 1, bucket_size)]
    return bucket_ranges


def get_padded_shapes(max_tokens=-1, max_characters=16, num_classes=2, num_tags=50, has_labels=True):
    """ Creates a dict of key: shape mappings for padding batches.

        @TODO This is a pretty ugly solution to support labelled/unlabelled modes, refactor target?

        Args:
            max_tokens: Max size of the context, -1 to pad to max within the batch.
            max_characters: Max number of characters, -1 to pad to max within the batch.
            num_classes: Number of classes in the dataset.
            num_tags: Number of Pos Tags in the dataset.
            has_labels: Include padded shape for answer_starts and answer_ends.
        Returns:
            A dict mapping of key: shape
    """
    shape_dict = {
        'words': [max_tokens],
        'chars': [max_tokens, max_characters],
        'tags': [max_tokens, num_tags],
        'num_tokens': []
    }

    if has_labels:
        shape_dict.update({
            'label': [num_classes],
        })

    return shape_dict


def create_lookup_tables(vocabs):
    """ Function that creates an index table for a word and character vocab, currently only works
        for vocabs without an explicit <PAD> character.
        Args:
            vocabs: List of strings representing a vocab for a table, string order in list determines lookup index.
        Returns:
            A lookup table for each vocab.
    """
    tables = []
    for vocab in vocabs:
        table = tf.contrib.lookup.index_table_from_tensor(mapping=tf.constant(vocab, dtype=tf.string),
                                                          default_value=len(vocab) - 1)
        tables.append(table)
    return tables


def create_pipeline(params, tables, record_paths, num_classes, num_tags, training=True):
    """ Function that creates an input pipeline for train/eval.

        Optionally uses bucketing to generate batches of a similar length. Output tensors
        are padded to the max within the batch.

        Args:
            params: A dictionary of parameters.
            tables: A tuple of contrib.lookup tables mapping string words to indices and string characters to indices.
            record_paths: A list of string filepaths for .tfrecord files.
            training: Boolean value signifying whether we are in train mode.
            num_classes: Number of classes in the dataset.
            num_tags: Number of POS tags in the dataset.
        Returns:
            A `tf.data.Dataset` object and an initializable iterator.
    """
    parallel_calls = get_num_parallel_calls(params)

    data = tf_record_pipeline(record_paths, params.tf_record_buffer_size, parallel_calls)
    data = data.cache()

    if training:
        data = data.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=params.shuffle_buffer_size))
    else:
        data = data.repeat()

    # Perform word -> index mapping.
    data = index_lookup(data, tables, char_limit=params.char_limit,
                        num_parallel_calls=parallel_calls)
    data = format_output(data, num_classes, num_tags, num_parallel_calls=parallel_calls)
    padded_shapes = get_padded_shapes(max_tokens=params.max_tokens, max_characters=params.char_limit,
                                      num_classes=num_classes, num_tags=num_tags)

    if params.bucket and training:
        buckets = create_buckets(params.bucket_size, params.max_tokens, params.bucket_ranges)

        def length_fn(fields):
            return fields['num_tokens']

        data = data.apply(
            tf.data.experimental.bucket_by_sequence_length(element_length_func=length_fn,
                                                           padded_shapes=padded_shapes,
                                                           bucket_batch_sizes=[params.batch_size] * (len(buckets) + 1),
                                                           bucket_boundaries=buckets))
    else:
        data = data.padded_batch(
            batch_size=params.batch_size,
            padded_shapes=padded_shapes,
            drop_remainder=training
        )

    data = data.prefetch(buffer_size=params.max_prefetch)
    iterator = data.make_initializable_iterator()
    return data, iterator


def create_demo_pipeline(params, tables, data):
    """ Function that creates an input pipeline for demo mode, .

        Output tensors are padded to the max within the batch.

        Args:
            params: A dictionary of parameters.
            tables: A tuple of contrib.lookup tables mapping string words to indices and string characters to indices.
            data: A dictionary containing keys for context_tokens, context_length, query_tokens, query_length and
                  answer_id.
        Returns:
            A `tf.data.Dataset` object and an initializable iterator.
    """
    parallel_calls = get_num_parallel_calls(params)

    data = tf.data.Dataset.from_tensor_slices(dict(data))
    data = index_lookup(data, tables, char_limit=params.char_limit,
                        num_parallel_calls=parallel_calls)
    padded_shapes = get_padded_shapes(max_characters=params.char_limit, has_labels=False)
    data = data.padded_batch(
        batch_size=params.batch_size,
        padded_shapes=padded_shapes,
        drop_remainder=False
    )
    data = data.prefetch(buffer_size=params.max_prefetch)
    iterator = data.make_initializable_iterator()
    return data, iterator


def create_placeholders():
    return {
        'tokens': tf.placeholder(shape=(None, None, ), dtype=tf.string, name='tokens'),
        'num_tokens': tf.placeholder(shape=(None, ), dtype=tf.int32, name='num_tokens'),
    }


def get_num_parallel_calls(params):
    """ Calculates the number of parallel calls we can make, if no number given returns the CPU count. """
    parallel_calls = params.parallel_calls
    if parallel_calls < 0:
        parallel_calls = os.cpu_count()
    return parallel_calls
