import pathlib
from tqdm import tqdm


def get_save_path(model, directory='./model_checkpoints', fold=None):
    model_name = model.__class__.__name__
    path = directory + '/{}/{}'.format(model_name, model_name)
    # create dirs if they don't exist.
    pathlib.Path(directory + '/{}/'.format(model_name)).mkdir(parents=True, exist_ok=True)

    if fold is not None:
        path = path + '-fold-{}'.format(fold)

    path = path + '.hdf5'

    return path


class CorpusStats:
    def __init__(self, corpus):
        self.corpus_stats = {}
        self.build_corpus_stats(corpus)

    def build_corpus_stats(self, corpus):
        print('Building Corpus Word Stats...')
        for entry in tqdm(corpus):
            cleaned = entry.split(' ')
            for word in cleaned:
                if word in self.corpus_stats:
                    self.corpus_stats[word] += 1
                else:
                    self.corpus_stats[word] = 1
