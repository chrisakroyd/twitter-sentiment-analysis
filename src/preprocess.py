import pandas as pd
from tqdm import tqdm
from src.util.sem_eval_util import concat_load_tsvs
from src.util.preprocessor import TextPreProcessor


def get_data_sent_140(path, max_examples):
    df = pd.read_csv(path, names=['class', 'id', 'date', 'query', 'user', 'text'], encoding='latin-1', index_col=1)
    df['class'] = df['class'].replace({0: 'negative', 4: 'positive'})
    df = df[:max_examples]
    return df


def get_data_sem_eval(data_dir):
    df = concat_load_tsvs(data_dir)
    return df


def pre_process(df):
    preprocessor = TextPreProcessor()
    new_df = pd.DataFrame.from_items([(name, pd.Series(data=None, dtype=series.dtype)) for name, series in df.iteritems()])
    rows = []
    # This is not my best work but it was the only way i could get it all working.
    # @TODO Revist with a fresh perspective.
    for index, tweet in tqdm(df.iterrows()):
        try:
            tweet['text'] = preprocessor.preprocess(tweet['text'])
            rows.append(tweet)
        except:
            pass

    new_df = new_df.append(rows)

    return new_df


def preprocess(hparams):
    if data_type == 'sem_eval':
        df = get_data_sem_eval(path)
    elif data_type == 'sent_140':
        df = get_data_sent_140(path)
    else:
        df = pd.read_csv(path, index_col=0)

    if print_classes:
        print(df['class'].value_counts())

    data_set = pre_process(df)

    tokenizer = Tokenizer(num_words=max_features, filters='"%()*+-:;=[\]^_`{|}~]')
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


