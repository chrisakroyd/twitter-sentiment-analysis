import os
from src import constants


def processed_data_directory(params):
    """ Generates a unique path to save processed data for a dataset """
    processed_data_dir = os.path.join(params.data_dir, constants.DirNames.PROCESSED, params.dataset)
    return processed_data_dir


def get_directories(params):
    """ Generates directory paths for data, processed data and saving models """
    return params.data_dir, processed_data_directory(params), params.models_dir


def raw_data_directory(params, dataset=None):
    """ Returns a path to the directory where the data for the given dataset is stored. """
    if dataset is None:
        dataset = params.dataset
    raw_data_dir = os.path.join(os.path.abspath(params.raw_data_dir), dataset)
    return raw_data_dir


def get_filenames(params):
    """ Gets the filenames for a specific dataset, if dataset isn't listed returns defaults.
        Args:
            params: A dictionary of parameters.
        returns:
            String paths for train + dev sets.
    """
    if params.dataset == constants.Datasets.SEM_EVAL:
        return constants.FilePaths.SEM_EVAL
    elif params.dataset == constants.Datasets.SEM_EVAL_2017:
        return constants.FilePaths.SEM_EVAL_2017
    elif params.dataset == constants.Datasets.SENT_140:
        return constants.FilePaths.SENT_140
    else:
        raise ValueError('Invalid dataset')


def raw_data_paths(params, dataset=None):
    """ Generates paths to raw data.
        Args:
            params: A dictionary of parameters.
            dataset: String dataset key.
        returns:
            String paths for raw squad train + dev sets.
    """
    if dataset is None:
        dataset = params.dataset

    raw_data_dir = raw_data_directory(params, dataset)
    train_name, dev_name = get_filenames(dataset)
    # Where we find the data
    train_path = os.path.join(raw_data_dir, train_name)
    dev_path = os.path.join(raw_data_dir, dev_name)

    return train_path, dev_path


def index_paths(params):
    """ Generates paths to word indexes.
        Args:
            params: A dictionary of parameters.
        returns:
            String paths for loading word, character and trainable indexes.
    """
    processed_dir = processed_data_directory(params)

    paths = []

    for embed_type in constants.EmbeddingTypes.as_list():
        paths += [os.path.join(processed_dir, constants.FileNames.INDEX.format(embedding_type=embed_type))]

    return paths


def embedding_paths(params):
    """ Generates paths to saved embedding files.
        Args:
            params: A dictionary of parameters.
        returns:
            String paths for loading word, character and trainable embeddings.
    """
    processed_dir = processed_data_directory(params)
    paths = []
    embed_types = [constants.EmbeddingTypes.WORD, constants.EmbeddingTypes.TRAINABLE, constants.EmbeddingTypes.CHAR]
    for embed_type in embed_types:
        paths += [
            os.path.join(processed_dir, constants.FileNames.EMBEDDINGS.format(embedding_type=embed_type))
        ]
    return paths


def save_paths(params):
    """ Generates paths to save trained models and logs for each run.
        Args:
            params: A dictionary of parameters.
        returns:
            String paths for loading data, saved models and saved logs.
    """
    model_path = os.path.join(params.models_dir, constants.DirNames.CHECKPOINTS, params.run_name)
    logs_path = os.path.join(params.models_dir, constants.DirNames.LOGS, params.run_name)
    return model_path, logs_path


def tf_record_paths(params):
    """ Generates a paths to .tfrecord files for train, dev and test.
        Args:
            params: A dictionary of parameters.
        returns:
            A string path to .tfrecord file.
    """
    processed_dir = processed_data_directory(params)

    paths = (
        os.path.join(processed_dir, constants.FileNames.TF_RECORD.format(name=constants.FileNames.TRAIN)),
        os.path.join(processed_dir, constants.FileNames.TF_RECORD.format(name=constants.FileNames.VAL)),
    )

    return paths


def examples_path(params):
    processed_dir = processed_data_directory(params)
    return os.path.join(processed_dir, constants.FileNames.EXAMPLES)


def meta_path(params):
    processed_dir = processed_data_directory(params)
    return os.path.join(processed_dir, constants.FileNames.META)


def classes_path(params):
    processed_dir = processed_data_directory(params)
    return os.path.join(processed_dir, constants.FileNames.CLASSES)
