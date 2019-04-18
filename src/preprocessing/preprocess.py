import random

import numpy as np
import pandas as pd
from tqdm import tqdm

from src import util, tokenizer as toke
from src import preprocessing as prepro
from sklearn.model_selection import train_test_split


def get_data_sent_140(path, max_examples=-1):
    df = pd.read_csv(path, names=['class', 'id', 'date', 'query', 'user', 'text'], encoding='latin-1', index_col=1)
    df['class'] = df['class'].replace({0: 'negative', 4: 'positive'})
    df = df[:max_examples]
    return df


def get_data_sem_eval(data_dir):
    df = util.concat_load_tsvs(data_dir)
    return df


def fit_and_extract(data_set, tokenizer, classes):
    tweets = []

    for index, tweet in tqdm(data_set.iterrows()):
        tweet_text = prepro.clean(tweet['text'])
        orig_tokens, modified_tokens, pos_tags = tokenizer.fit_on_texts(tweet_text)[-1]
        num_tokens = len(modified_tokens)

        if num_tokens > 0:
            tweets.append({
                'text': tweet['text'],
                'orig_tokens': orig_tokens,
                'tokens': modified_tokens,
                'tags': pos_tags,
                'num_tokens': num_tokens,
                'label': classes[tweet['class']],
            })

    return tweets, tokenizer


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

    tweets, tokenizer = fit_and_extract(data, tokenizer, classes)

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

    writer = prepro.RecordWriter(params.max_tokens)
    writer.write(train_record_path, train)
    writer.write(val_record_path, val)

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
