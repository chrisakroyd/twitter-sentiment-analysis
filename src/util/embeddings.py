import numpy as np
from .util import load_json


def create_vocab(embedding_index):
    return set([e for e, _ in embedding_index.items()])


def generate_matrix(index, embedding_dimensions=300, skip_zero=True):
    rows = len(index) + 1 if skip_zero else len(index)
    matrix = np.random.normal(scale=0.1, size=(rows, embedding_dimensions))

    if skip_zero:
        matrix[0] = np.zeros(embedding_dimensions)

    return matrix


def generate_trainable_matrix(embedding_index, word_index, embedding_dimensions, trainable_words):
    trainable_matrix = np.zeros((len(trainable_words) + 1, embedding_dimensions))
    valid_word_range = len(word_index) - len(trainable_words)

    for word in trainable_words:
        if word in word_index:
            # Zero out the vector for this word in the pre-trained index.
            embedding_index[word] = np.zeros((embedding_dimensions, ))
            # Randomly initialise the trainable word.
            trainable_matrix[(word_index[word] - valid_word_range)] = np.random.normal(scale=0.1,
                                                                                       size=(embedding_dimensions, ))

    return trainable_matrix, embedding_index


def read_embeddings_file(path):
    embedding_index = {}
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            values = line.strip().split(' ')
            # First line is num words + vector size when using fast text, we skip this.
            if i == 0 and len(values) == 2:
                print('Detected FastText vector format.')
                continue
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coefs

    return embedding_index


def load_embedding_matrix(embedding_index, word_index, embedding_dimensions):
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dimensions))

    for word, index in word_index.items():
        if index > len(embedding_matrix):
            raise ValueError('Index larger than embedding matrix for {}'.format(word))

        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[index] = embedding_vector

    return embedding_matrix


def load_embedding(path,
                   word_index,
                   embedding_dimensions=300,
                   trainable_embeddings=[],
                   embedding_index=None):
    # Read the given embeddings file if its not given.
    embedding_index = embedding_index if embedding_index is not None else read_embeddings_file(path)

    if len(trainable_embeddings) > 0:
        trainable_matrix, embedding_index = generate_trainable_matrix(embedding_index,
                                                                      word_index,
                                                                      embedding_dimensions,
                                                                      trainable_embeddings)
    else:
        trainable_matrix = np.zeros((1, embedding_dimensions))

    embedding_matrix = load_embedding_matrix(embedding_index, word_index, embedding_dimensions)

    return embedding_matrix, trainable_matrix


def save_embeddings(path, embedding_matrix, word_index):
    with open(path, 'w', encoding='utf8') as embeddings:
        for key, value in word_index.items():
            embed = embedding_matrix[value, :]
            embeddings.write('%s %s\n' % (key, ' '.join(map(str, embed))))


def merge_embeddings(embedding_matrix, trainable_embedding_matrix, word_index, trainable_words):
    vocab_size = len(word_index) + 1
    num_trainable = len(trainable_words) + 1
    valid_word_range = vocab_size - num_trainable

    for word in trainable_words:
        index = word_index[word]
        # Merge the trainable and embedding matrix
        embedding_matrix[index] = trainable_embedding_matrix[index - valid_word_range]

    return embedding_matrix


def load_embeddings(index_paths, embedding_paths, embed_dim, char_dim):
    word_index_path, trainable_index_path, char_index_path = index_paths
    word_embedding_path, trainable_embedding_path, char_embedding_path = embedding_paths

    print('Loading Embeddings...')

    word_index = load_json(word_index_path)
    trainable_word_index = load_json(trainable_index_path)
    char_index = load_json(char_index_path)

    word_matrix, _ = load_embedding(word_embedding_path, word_index, embed_dim)
    trainable_matrix, _ = load_embedding(trainable_embedding_path, trainable_word_index, embed_dim)
    character_matrix, _ = load_embedding(char_embedding_path, char_index, char_dim)

    return word_matrix, trainable_matrix, character_matrix
