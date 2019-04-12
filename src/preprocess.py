import random

import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm

from src import util, tokenizer as toke
from src.preprocessor import TextPreProcessor
from sklearn.model_selection import train_test_split


def get_data_sent_140(path, max_examples=-1):
    df = pd.read_csv(path, names=['class', 'id', 'date', 'query', 'user', 'text'], encoding='latin-1', index_col=1)
    df['class'] = df['class'].replace({0: 'negative', 4: 'positive'})
    df = df[:max_examples]
    return df


def get_data_sem_eval(data_dir):
    df = util.concat_load_tsvs(data_dir)
    return df


def fit_and_extract(data_set, tokenizer, classes, params):
    preprocessor = TextPreProcessor()
    tweets = []

    for index, tweet in tqdm(data_set.iterrows()):
        tweet_text = preprocessor.preprocess(tweet['text'])
        _, modified_tokens, pos_tags = tokenizer.fit_on_texts(tweet_text)[-1]
        num_tokens = len(modified_tokens)

        if 0 < num_tokens < params.max_tokens:
            tweets.append({
                'text': tweet['text'],
                'tokens': modified_tokens,
                'tags': pos_tags,
                'num_tokens': num_tokens,
                'label': classes[tweet['class']],
            })

    return tweets, tokenizer


def write_as_tf_record(path, data):
    """ Shuffles the queries and writes out the context + queries as a .tfrecord file.

        Args:
            path: Output path for the .tfrecord file.
            data:
    """
    records = []
    shuffled = list(data)
    random.shuffle(shuffled)
    for data in shuffled:
        encoded_tokens = [m.encode('utf-8') for m in data['tokens']]
        encoded_tags = [m.encode('utf-8') for m in data['tags']]

        tokens = tf.train.Feature(bytes_list=tf.train.BytesList(value=encoded_tokens))
        tags = tf.train.Feature(bytes_list=tf.train.BytesList(value=encoded_tags))
        num_tokens = tf.train.Feature(int64_list=tf.train.Int64List(value=[data['num_tokens']]))
        label = tf.train.Feature(int64_list=tf.train.Int64List(value=[data['label']]))

        record = tf.train.Example(features=tf.train.Features(feature={
            'tokens': tokens,
            'tags': tags,
            'num_tokens': num_tokens,
            'label': label,
        }))

        records.append(record)

    with tf.python_io.TFRecordWriter(path) as writer:
        for record in records:
            writer.write(record.SerializeToString())


def get_examples(tweets, num_examples=1000):
    shuffled = tweets
    random.shuffle(shuffled)
    shuffled = shuffled[:num_examples]
    return shuffled


def process(params, data, print_classes=True):
    directories = util.get_directories(params)
    util.make_dirs(directories)
    # path to save tf_records and a random sample of data.
    train_record_path, val_record_path = util.tf_record_paths(params)
    examples_path = util.examples_path(params)
    meta_path = util.meta_path(params)
    classes_path = util.classes_path(params)
    # Get paths for saving embedding related info.
    word_index_path, trainable_index_path, char_index_path, pos_index_path = util.index_paths(params)
    word_embeddings_path, trainable_embeddings_path, char_embeddings_path = util.embedding_paths(params)

    if print_classes:
        print(data['class'].value_counts())

    # Read the embedding index and create a vocab of words with embeddings.
    print('Loading Embeddings, this may take some time...')
    embedding_index = util.read_embeddings_file(params.embeddings_path)
    vocab = set([e for e, _ in embedding_index.items()])

    tokenizer = toke.Tokenizer(max_words=params.max_words + 1,
                               max_chars=params.max_chars + 1,
                               vocab=vocab,
                               lower=False,
                               oov_token=params.oov_token,
                               min_word_occurrence=params.min_word_occur,
                               min_char_occurrence=params.min_char_occur,
                               trainable_words=params.trainable_words,
                               filters=None)

    classes = data['class'].unique()
    classes = util.index_from_list(classes, skip_zero=False)

    tweets, tokenizer = fit_and_extract(data, tokenizer, classes, params)

    tokenizer.init()
    word_index = tokenizer.word_index
    char_index = tokenizer.char_index
    trainable_index = util.index_from_list(params.trainable_words)

    embedding_matrix = util.load_embedding_file(path=params.embeddings_path,
                                                word_index=word_index,
                                                embedding_dimensions=params.embed_dim,
                                                trainable_embeddings=params.trainable_words,
                                                embedding_index=embedding_index)

    trainable_matrix = util.generate_matrix(index=trainable_index, embedding_dimensions=params.embed_dim)
    char_matrix = util.generate_matrix(index=char_index, embedding_dimensions=params.char_dim)

    print('Number of Data Samples:' + str(len(tweets)))

    train, val = train_test_split(tweets, test_size=0.2)

    num_classes = len(classes)
    print('Num classes: ' + str(num_classes))

    write_as_tf_record(train_record_path, train)
    write_as_tf_record(val_record_path, val)
    examples = get_examples(train)
    meta = {
        'num_train': len(train),
        'num_val': len(val),
        'num_classes': num_classes,
        'num_tags': len(tokenizer.tag_index)
    }

    # Save meta information, e.g. number of train/val/classes
    util.save_json(meta_path, meta)
    util.save_json(classes_path, classes)
    # Save a random sample of the data.
    util.save_json(examples_path, examples)
    # Save the word index mapping of word:index for both the pre-trained and trainable embeddings.
    util.save_json(word_index_path, word_index)
    util.save_json(char_index_path, char_index)
    util.save_json(trainable_index_path, trainable_index)
    util.save_json(pos_index_path, tokenizer.tag_index)
    # Save the trainable embeddings matrix.
    np.save(trainable_embeddings_path, trainable_matrix)
    # Save the full embeddings matrix
    np.save(word_embeddings_path, embedding_matrix)
    np.save(char_embeddings_path, char_matrix)
