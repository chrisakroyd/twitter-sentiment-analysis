import keras.backend as K
import os
import sys

# Only use the amount of memory we require rather than the maximum
if 'tensorflow' == K.backend():
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = "0"
    set_session(tf.Session(config=config))

module_path = os.path.abspath(os.path.join('./src/neural_backend'))

print(module_path)

if module_path not in sys.path:
    sys.path.append(module_path)

from keras.callbacks import ModelCheckpoint, EarlyStopping
# Utility code.
from src.neural_backend.load_data import load_data
from src.neural_backend.load_embeddings import load_embeddings, load_afinn_matrix
from src.neural_backend.preprocessor import create_polarity_dict
from src.neural_backend.util import get_save_path
# Models
from src.neural_backend.models.lstm_attention import BiLSTMAttention

TRAIN = True
PRODUCTION = True
WRITE_RESULTS = True
MAX_FEATS = 150000

# Paths to data sets
sent_140_path = './data/sent_140/training.1600000.processed.noemoticon.csv'
sem_eval_path = './data/sem_eval/full/'
sem_eval_2017_path = './data/sem_eval/2017_dataset/'
# Paths to glove embeddings.
glove_path = './data/embeddings/glove.twitter.27B.200d.txt'
embed_dims = 200
embed_type = 'GLOVE'


# (x_train, y_train), (x_val, y_val), word_index, num_classes = load_data(path=sem_eval_2017_path,
#                                                            data_set='sem_eval',
#                                                            max_features=MAX_FEATS)
#
(x_train, y_train), (x_val, y_val), word_index, num_classes = load_data(path=sem_eval_path,
                                                           data_set='sem_eval',
                                                           max_features=MAX_FEATS)
# #
# (x_train, y_train), (x_val, y_val), word_index, num_classes = load_data(path=sent_140_path,
#                                                            data_set='sent_140',
#                                                            max_features=MAX_FEATS)

embedding_matrix = load_embeddings(path=glove_path,
                                   embedding_type=embed_type,
                                   word_index=word_index,
                                   max_features=MAX_FEATS,
                                   embedding_dimensions=embed_dims)

polarity = create_polarity_dict()
afinn_matrix = load_afinn_matrix(word_index, polarity)

vocab_size = len(word_index) + 1

# model_instance = BiLSTMConcPool(num_classes=num_classes)
model_instance = BiLSTMAttention(num_classes=num_classes)
# model_instance = BiLSTMAttentionSkip(num_classes=num_classes)

print(num_classes)

if TRAIN:

    print(x_train.shape)
    model = model_instance.build(vocab_size,
                                 embedding_matrix,
                                 afinn_matrix,
                                 input_length=x_train.shape[1],
                                 embed_dim=embed_dims)

    checkpoint = ModelCheckpoint(get_save_path(model_instance), save_best_only=True)

    early_stop = EarlyStopping(monitor='val_loss',
                               patience=5,
                               verbose=1,
                               min_delta=0.00001)

    model.fit(x=x_train,
              y=y_train,
              validation_data=(x_val, y_val),
              epochs=model_instance.EPOCHS,
              batch_size=model_instance.BATCH_SIZE,
              callbacks=[checkpoint, early_stop])

elif PRODUCTION:
    model = get_save_path(model_instance)
