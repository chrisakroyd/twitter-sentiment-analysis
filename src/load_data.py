import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle

# Tokenize and sequence padding.
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

# Semeval tsv loader
from .sem_eval_utilities import concat_load_tsvs

RANDOM_SEED = 59185
CONTROL_BALANCE = True
DATASET_SIZE = 250000
# DATASET_SIZE = 50
SEQUENCE_LENGTH = 75

mentions_regex = re.compile('(?<=^|(?<=[^a-zA-Z0-9-_.]))@([A-Za-z_]+[A-Za-z0-9_]+)')
hash_tag_regex = re.compile('(?<=^|(?<=[^a-zA-Z0-9-_.]))#([A-Za-z]+[A-Za-z0-9]+)')
url_regex = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
# Smile, Laugh, Love, Wink emoticon regex : :), : ), :-), (:, ( :, (-:, :'), :D, : D, :-D, xD, x-D, XD, X-D,
# <3, ;-), ;), ;-D, ;D, (;, (-;
emo_pos_regex = re.compile('(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))|(:\s?D|:-D|x-?D|X-?D)|(<3)|(;-?\)|;-?D|\(-?;)')
# Sad & Cry emoticon regex: :-(, : (, :(, ):, )-:, :,(, :'(, :"(
emo_neg_regex = re.compile('(:\s?\(|:-\(|\)\s?:|\)-:)|(:,\(|:\'\(|:"\()')

collapse_letters_regex = re.compile('(.)\1+')


def get_data_sent_140(path):
    df = pd.read_csv(path, names=['class', 'id', 'date', 'query', 'user', 'text'], encoding='latin-1')
    df['class'] = df['class'].replace({0: 'negative', 4: 'positive'})
    print(df['class'].value_counts())
    df = shuffle(df)
    # return df[:DATASET_SIZE]
    return df


def get_data_sem_eval(data_dir):
    df = concat_load_tsvs(data_dir)
    print(df['class'].value_counts())
    return df


def pre_process(df):
    new_df = pd.DataFrame.from_items([(name, pd.Series(data=None, dtype=series.dtype)) for name, series in df.iteritems()])
    rows = []

    FLAGS = re.MULTILINE | re.DOTALL

    def hashtag(text):
        text = text.group()
        hashtag_body = text[1:]
        if hashtag_body.isupper():
            result = "<hashtag> {} <allcaps>".format(hashtag_body)
        else:
            result = " ".join(["<hashtag>"] + re.split(r"(?=[A-Z])", hashtag_body, flags=FLAGS))
        return result

    def allcaps(text):
        text = text.group()
        return text.lower() + " <allcaps>"

    def tokenize(text):
        # Different regex parts for smiley faces
        eyes = r"[8:=;]"
        nose = r"['`\-]?"

        # function so code less repetitive
        def re_sub(pattern, repl):
            return re.sub(pattern, repl, text, flags=FLAGS)

        text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
        text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>")
        text = re_sub(r"{}{}p+".format(eyes, nose), "<lolface>")
        text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>")
        text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>")
        text = re_sub(r"/", " / ")
        text = re_sub(r"@\w+", "<user>")
        text = re_sub(r"<3", "<heart>")
        text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
        text = re_sub(r"#\S+", hashtag)
        text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
        text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")

        text = mentions_regex.sub('<mention>', text)
        text = re_sub(r"([A-Z]){2,}", allcaps)

        return text.lower()

    # This is not my best work but it was the only way i could get it all working.
    # @TODO Revist with a fresh perspective.
    for index, tweet in df.iterrows():
        try:
            tweet['text'] = tokenize(tweet['text'])
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

    X = pad_sequences(tokenizer.texts_to_sequences(data_set['text']), maxlen=SEQUENCE_LENGTH)
    y = label_binarizer.fit_transform(data_set['class'])

    if len(label_binarizer.classes_) == 2:
        y = np.hstack((y, 1 - y))

    print('Number of Data Samples:' + str(len(X)))

    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)
    num_classes = len(label_binarizer.classes_)

    return (np.array(x_train), np.array(y_train)), (np.array(x_val), np.array(y_val)), word_index, num_classes
