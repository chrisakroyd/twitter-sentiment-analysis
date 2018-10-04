import pandas as pd
import glob


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

    for counter, file in enumerate(glob.glob(data_dir + "*.tsv")):
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
