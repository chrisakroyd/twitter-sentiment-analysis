from src.config import model_config
from src.constants import FilePaths
from src.preprocess import get_data_sem_eval, get_data_sent_140, preprocess
from src.util import namespace_json, get_directories, make_dirs


def preprocess(hparams):
    dataset = hparams.dataset.lower()
    make_dirs(get_directories(hparams))

    if dataset == 'semeval':
        data = get_data_sem_eval(path)
    elif dataset == 'sent_140':
        data = get_data_sent_140(path)
    else:
        raise NotImplementedError('Unsupported dataset: Valid datasets are {}.'.format('squad'))

    data_set = preprocess(data)

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

if __name__ == '__main__':
    defaults = namespace_json(path=FilePaths.defaults.value)
    preprocess(model_config(defaults))
