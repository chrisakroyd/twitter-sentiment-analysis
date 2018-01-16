import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle

# Tokenize and sequence padding.
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

RANDOM_SEED = 59185
CONTROL_BALANCE = True
DATASET_SIZE = 50000

mentions_regex = re.compile('(?<=^|(?<=[^a-zA-Z0-9-_.]))@([A-Za-z_]+[A-Za-z0-9_]+)')
hash_tag_regex = re.compile('(?<=^|(?<=[^a-zA-Z0-9-_.]))#([A-Za-z]+[A-Za-z0-9]+)')
url_regex = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')


def get_data_sent_140(path):
    df = pd.read_csv(path, names=['class', 'id', 'date', 'query', 'user', 'text'], encoding='latin-1')
    df = shuffle(df)
    return df[:DATASET_SIZE]


def get_data_sem_eval(path):
    df = pd.read_csv(path, names=['id', 'class', 'text', 'bl'], sep='\t')
    df = df.drop(columns=['bl'])
    return df


def pre_process(df):
    new_df = pd.DataFrame.from_items([(name, pd.Series(data=None, dtype=series.dtype)) for name, series in df.iteritems()])
    rows = []

    # This is not my best work but it was the only way i could get it all working.
    # @TODO Revist with a fresh perspective.
    for index, tweet in df.iterrows():
        try:
            tweet['text'] = mentions_regex.sub('<MENTION>', tweet['text'])
            tweet['text'] = hash_tag_regex.sub('<HASH_TAG>', tweet['text'])
            tweet['text'] = url_regex.sub('<URL>', tweet['text'])
            rows.append(tweet)
        except:
            pass

    new_df = new_df.append(rows)

    return new_df


def load_data(path, data_set='sem_eval', max_features=5000,):
    if data_set == 'sem_eval':
        df = get_data_sem_eval(path)
    elif data_set == 'sent_140':
        df = get_data_sent_140(path)
    else:
        print('INVALID DATA GENERATOR SPECIFIED')

    data_set = pre_process(df)

    tokenizer = Tokenizer(num_words=max_features)
    label_binarizer = LabelBinarizer()
    tokenizer.fit_on_texts(data_set.text)
    word_index = tokenizer.word_index

    #
    X = pad_sequences(tokenizer.texts_to_sequences(data_set['text']))
    y =label_binarizer.fit_transform(data_set['class'])

    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

    return (np.array(x_train), np.array(y_train)), (np.array(x_val), np.array(y_val)), word_index
