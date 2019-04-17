import json
import glob
import os
from collections import ChainMap
from types import SimpleNamespace
import pandas as pd
from tqdm import tqdm
from src import constants, util


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


def save_json(path, data, indent=None):
    """ Saves data as a UTF-8 encoded .json file.
        Args:
            path: String path to a .json file.
            data: A dict or iterable.
            indent: Pretty print the json with this level of indentation.
    """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent)


def load_json(path):
    """ Loads a UTF-8 encoded .json file.
        Args:
            path: String path to a .json file.
        Returns:
            Loaded json as original saved type e.g. dict for index, list for saved lists.
    """
    assert isinstance(path, str) and len(path) > 0
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def load_multiple_jsons(paths):
    """ Loads multiple UTF-8 encoded .json file.
        Args:
            paths: List of string paths to .json files.
        Returns:
            Loaded json as original saved type e.g. dict for index, list for saved lists for each path.
    """
    assert isinstance(paths, list) or isinstance(paths, tuple)
    return [load_json(path) for path in paths]


def namespace_json(path):
    """ Turns a dict into an object, allows lookup via dot notation.
        Args:
            path: String path to a .json file.
        Returns:
            A namespace object.
    """
    return SimpleNamespace(**load_json(path))


def params_as_dict(params):
    """
        Converts all flags + values generated with the abseil-py flag (tf.flags) object into a flat python dict.
        Args:
            params: An instance of flags.Flags
        Returns:
            A flat dict of all flag parameters.
    """
    # Flags are segregated by the file they were defined in, reduce this down to a flat list of all our flags.
    flag_modules = [{v.name: v.value for v in values} for _, values in params.flags_by_module_dict().items()]
    full_flags = dict(ChainMap(*flag_modules))  # ChainMap == dictionary update within loop
    return full_flags


def load_config(params, path):
    """ Loads a config if it exists and there's existing checkpoints, otherwise alerts the user and returns params."""
    if file_exists(path):
        if len(os.listdir(os.path.dirname(path))) == 1:  # Only have a model config, check if this is intentional.
            if not util.yes_no_prompt(constants.Prompts.FOUND_CONFIG_NO_CHECKPOINTS.format(path=path)):
                os.remove(path)  # Delete the config.
                return params
        print('Using config for {run_name} found at {path}...'.format(run_name=params.run_name, path=path))
        return namespace_json(path)
    else:
        print('No existing config for {run_name}...'.format(run_name=params.run_name))
        return params


def save_config(params, path, overwrite=False):
    """ Saves abseil-py flag (tf.flags) object as a .json formatted file. """
    if not file_exists(path) or overwrite:
        save_json(path, params_as_dict(params), indent=2)


def index_from_list(words, skip_zero=True):
    """ Turns a list of strings into a word: index lookup table.
        Args:
            words: A list of strings.
            skip_zero: Whether 0 should be skipped.
        Returns:
            A dict mapping words to integers.
    """
    return {word: (i+1) if skip_zero else i for i, word in enumerate(words)}


def file_exists(path):
    """ Tests whether or not a full filepath exists. """
    return os.path.exists(path)


def directory_exists(path):
    """ Tests whether or not a directory exists. """
    return os.path.isdir(path)


def directory_is_empty(path):
    """ Tests whether or not a directory is empty. """
    return len(os.listdir(path)) == 0


def make_dirs(directories):
    """ Creates non-existent directories.
        Args:
            directories: A string directory path or a list of directory paths.
    """
    if isinstance(directories, str):
        directories = [directories]

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)


def load_vocab_files(paths):
    """ Loads a .json index as a list of words where each words position is its index.
        Args:
            paths: Iterable of string paths or string path pointing to .json word index file.
        Returns:
            A list of strings.
    """
    if isinstance(paths, str):
        paths = [paths]

    vocabs = []
    for path in paths:
        index = load_json(path)
        vocabs.append(sorted(index, key=index.get))
    return vocabs


def remove_keys(data, keys=[]):
    """ Removes specified keys from a list of dicts.
        Args:
            data: Iterable of dicts.
            keys: List of string keys to remove.
        Returns:
            Input data with keys removed.
    """
    for _, value in data.items():
        for key in keys:
            value.pop(key, None)
    return data


def load_sem_eval_2017_txt(path, save_path=None):
    """
    Semeval's provided input data is of a strange formatting that requires an extra column when loaded into panda's.
    This script simply loads the txt and formats it as a tsv so no pandas blank column is required to
    load the correct data.

    - Implementation comments
    * The reason for using TSV rather than CSV is because all the previous years Sem Eval stuff is TSV files.

    :param path: A filepath to a semeval train/test .txt file.
    :param save_path: An optional file path to save the correctly formatted TSV data to.
    :return: True/False: Whether we successfully save the file.
    """
    df = pd.read_csv(path, names=['id', 'class', 'text', 'bl'], sep='\t', index_col=0)
    df = df.drop(columns=['bl'])

    if save_path:
        df.to_csv(save_path, sep='\t', header=False)

    return df


def concat_load_tsvs(data_dir, save_path=None):
    """
    Loads a series of TSV's to form one complete data set, this is performed as the 'full' sem eval dataset
    is spread over multiple years of challenges. Therefore we supplement the 2017 data with the previous years
    challenge data.

    - Implementation comments
    * The reason for using TSV rather than CSV is because all the previous years Sem Eval stuff is TSV files.

    :param data_dir:
    :param save_path:
    :return:
    """
    full_data_set = pd.DataFrame()

    for counter, file in enumerate(glob.glob(data_dir + "/*.tsv")):
        # Read the tsvs generated from the previous years script https://github.com/seirasto/twitter_download
        # and the pre-downloaded version from the 2017 txt file formatted as a CSV.
        curr_file = pd.read_csv(file, names=['id', 'class', 'text'], sep='\t', encoding='utf-8')
        full_data_set = full_data_set.append(curr_file)

    full_data_set = full_data_set.set_index('id')
    full_data_set = full_data_set.drop_duplicates('text')

    full_data_set['text'] = full_data_set['text'].apply(lambda x: x.encode('utf-8').decode('raw_unicode_escape'))
    full_data_set['text'] = full_data_set['text'].apply(lambda x: x.strip())

    if save_path:
        full_data_set.to_csv(save_path, sep='\t', header=False)

    return full_data_set
