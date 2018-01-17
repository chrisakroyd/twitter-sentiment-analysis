import time
import keras.backend as K

# Only use the amount of memory we require rather than the maximum
if 'tensorflow' == K.backend():
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = "0"
    set_session(tf.Session(config=config))

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import load_model

# Utility code.
from src.load_data import load_data
from src.load_glove_embeddings import load_embedding_matrix
from src.models.bi_lstm import BiLSTMModel
from src.models.bi_lstm_attention import BiLSTMAttention
from src.models.cnn import CNNModel

TRAIN = True
PRODUCTION = True
WRITE_RESULTS = True
MAX_FEATS = 5000

# Paths to data sets
tweet_path = './data/training.1600000.processed.noemoticon.csv'
sem_eval_path = './data/SemEval2017-task4-dev.subtask-A.english.INPUT.txt'
# Paths to glove embeddings.
glove_path = './data/glove.twitter.27B/glove.twitter.27B.100d.txt'
glove_embed_dims = 100


(x_train, y_train), (x_val, y_val), word_index = load_data(path=sem_eval_path,
                                                           data_set='sem_eval',
                                                           max_features=MAX_FEATS)

embedding_matrix = load_embedding_matrix(glove_path=glove_path,
                                         word_index=word_index,
                                         embedding_dimensions=glove_embed_dims)

vocab_size = len(word_index) + 1

# model_instance = CNNModel(num_classes=3)
model_instance = BiLSTMAttention(num_classes=3)

if TRAIN:

    print(x_train.shape)
    model = model_instance.create_model(vocab_size,
                                        embedding_matrix,
                                        input_length=x_train.shape[1],
                                        embed_dim=glove_embed_dims)

    tensorboard = TensorBoard(log_dir='./logs/{}'.format(time.time()), batch_size=model_instance.BATCH_SIZE)
    checkpoint = ModelCheckpoint(model_instance.checkpoint_path, monitor='val_loss')
    early_stop = EarlyStopping(monitor='val_loss',
                               patience=12,
                               verbose=1,
                               min_delta=0.00001)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                  patience=2,
                                  verbose=1,
                                  epsilon=0.0001,
                                  mode='min', min_lr=0.0001)

    model.fit(x=x_train,
              y=y_train,
              validation_data=(x_val, y_val),
              epochs=model_instance.EPOCHS,
              batch_size=model_instance.BATCH_SIZE,
              callbacks=[tensorboard, checkpoint, early_stop])

elif PRODUCTION:
    model = load_model(model_instance.checkpoint_path)
