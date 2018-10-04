import keras.backend as K

# Only use the amount of memory we require rather than the maximum
if 'tensorflow' == K.backend():
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = "0"
    set_session(tf.Session(config=config))

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
# Utility code.
from src.util.load_data import load_data
from src.util import get_save_path, namespace_json, load_embeddings
from src.constants import FilePaths
from src.config import model_config

# Models
from src.models.lstm_attention import BiLSTMAttention

MAX_FEATS = 150000

# Paths to data sets
sent_140_path = './data/sent_140/training.1600000.processed.noemoticon.csv'
sem_eval_path = './data/sem_eval/full/'
sem_eval_2017_path = './data/sem_eval/2017_dataset/'

# Paths to glove embeddings.
glove_path = './data/embeddings/glove.twitter.27B.200d.txt'
embed_dims = 200
embed_type = 'GLOVE'


def train(hparams):
    (x_train, y_train), (x_val, y_val), word_index, num_classes, lb, tokenizer = load_data(path=sem_eval_path,
                                                                                           data_type='sem_eval',
                                                                                           max_features=MAX_FEATS)
    # #
    # (x_train, y_train), (x_val, y_val), word_index, num_classes, lb, tokenizer = load_data(path=sent_140_path,
    #                                                            data_type='sent_140',
    #                                                            max_features=MAX_FEATS)

    embedding_matrix = load_embeddings(path=hparams.embed_path,
                                       word_index=word_index,
                                       embedding_dimensions=hparams.embed_dims)

    vocab_size = len(word_index) + 1

    # model_instance = BiLSTMAttentionSkip(num_classes=num_classes)
    model_instance = BiLSTMAttention(num_classes=num_classes)

    print(num_classes)

    print(x_train.shape)
    model = model_instance.build(vocab_size,
                                 embedding_matrix,
                                 input_length=x_train.shape[1],
                                 embed_dim=embed_dims)

    checkpoint = ModelCheckpoint(get_save_path(model_instance), save_best_only=True)

    early_stop = EarlyStopping(monitor='val_loss',
                               patience=5,
                               verbose=1,
                               min_delta=0.00001)

    logging = TensorBoard(log_dir='./data/logs')

    model.fit(x=x_train,
              y=y_train,
              validation_data=(x_val, y_val),
              epochs=hparams.epochs,
              batch_size=hparams.batch_size,
              callbacks=[checkpoint, early_stop, logging])


if __name__ == '__main__':
    defaults = namespace_json(path=FilePaths.defaults.value)
    train(model_config(defaults))
