import numpy as np
from tqdm import tqdm


def add_custom_embeddings(embedding_index):
    dimension = len(embedding_index[list(embedding_index.keys())[0]])
    # Empty tag is only introduced when there is no text - Should not occur
    embedding_index['<empty>'] = np.full((dimension,), 0.999)
    # Add a vector for my custom tags - at least until own embeddings are trained.
    embedding_index['<ip>'] = np.full((dimension,), 0.155)
    embedding_index['<date>'] = np.full((dimension, ), 0.145)
    embedding_index['<percent>'] = np.full((dimension,), 0.135)
    embedding_index['<score>'] = np.full((dimension,), 0.115)
    embedding_index['<currency>'] = np.full((dimension,), 0.105)
    embedding_index['<email>'] = np.full((dimension,), 0.095)
    embedding_index['<kisses>'] = np.full((dimension,), 0.085)
    embedding_index['<emphasis>'] = np.full((dimension,), 0.075)

    if '<hashtag>' in embedding_index:
        embedding_index['</hashtag>'] = embedding_index['<hashtag>']
    else:
        embedding_index['<hashtag>'] = np.full((dimension, ), 0.125)
        embedding_index['</hashtag>'] = np.full((dimension, ), 0.125)

    return embedding_index


def read_embeddings_file(path, embedding_type):
    embedding_index = {}
    print('Reading {} file...'.format(embedding_type))
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in tqdm(enumerate(f)):
            # First line is num words/vector size.
            if i == 0 and embedding_type == 'FAST_TEXT':
                continue
            values = line.strip().split(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coefs

    embedding_index = add_custom_embeddings(embedding_index)

    return embedding_index


def load_embedding_matrix(embedding_index, word_index, max_features, embedding_dimensions, rand_oov=True):
    print('Generating embedding matrix...')
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dimensions))
    oov_count = 0
    for word, i in tqdm(word_index.items()):
        if i > max_features: continue
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            # Give OOV's an embedding for the word 'something' until i bother to train my own embeddings with an <OOV>
            # character
            embedding_matrix[i] = embedding_index.get('something')

            oov_count += 1

    print('{} Unique words (vocab size).'.format(len(embedding_matrix)))
    print('{} Unique OOV words.'.format(oov_count))

    return embedding_matrix, oov_count


def load_afinn_matrix(word_index, polarity_dict):
    print('Generating embedding matrix...')
    embedding_matrix = np.zeros((len(word_index) + 1, 1))
    for word, i in tqdm(word_index.items()):
        embedding_vector = polarity_dict.get(word)
        if embedding_vector is not None:
            # words not found in afinn will be all-zeros.
            embedding_matrix[i] = np.array(embedding_vector)

    return embedding_matrix


def load_embeddings(path, embedding_type, word_index, max_features, embedding_dimensions=300, embedding_index=None):
    if embedding_type == 'GLOVE' or embedding_type == 'FAST_TEXT':
        print('Loading {} embeddings...'.format(embedding_type))
    else:
        print('Generating random uniform embeddings...')
        return np.random.uniform(low=-0.05, high=0.05, size=(len(word_index) + 1, embedding_dimensions))

    if embedding_index is None:
        embedding_index = read_embeddings_file(path, embedding_type)

    embedding_matrix, oov_count = load_embedding_matrix(embedding_index, word_index, max_features, embedding_dimensions)

    return embedding_matrix

