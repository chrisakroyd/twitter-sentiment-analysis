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

# Utility code.
from src.neural_backend.load_data import load_data, get_data_sem_eval
from src.neural_backend.load_embeddings import load_embeddings
from src.neural_backend.layers.Attention import FeedForwardAttention as Attention
from src.neural_backend.metrics import f1, precision, recall
from src.neural_backend.preprocessor import TextPreProcessor

from keras.layers import Input, Dense, Bidirectional, Dropout, LSTM, Embedding, SpatialDropout1D, GaussianNoise
from keras.regularizers import l2
from keras.optimizers import RMSprop
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences

from flask import Flask, json, request

app = Flask(__name__)

MAX_FEATS = 150000

# Paths to data sets
sem_eval_path = './data/sem_eval/full/'
# Paths to glove embeddings.
glove_path = './data/embeddings/glove.twitter.27B.200d.txt'
embed_dims = 200
embed_type = 'GLOVE'
BATCH_SIZE = 128
EPOCHS = 50
LEARN_RATE = 0.001
CLIP_NORM = 5.0
NUM_CLASSES = 12
RNN_UNITS = 150
L2_REG = 0.0001
SEQUENCE_LENGTH = 40

preprocessor = TextPreProcessor()

sem_eval = get_data_sem_eval(sem_eval_path)

(x_train, y_train), (x_val, y_val), word_index, num_classes, lb, tokenizer = load_data(path=sem_eval_path,
                                                           data_set='sem_eval',
                                                           max_features=MAX_FEATS)
embedding_matrix = load_embeddings(path=glove_path,
                                   embedding_type=embed_type,
                                   word_index=word_index,
                                   max_features=MAX_FEATS,
                                   embedding_dimensions=embed_dims)

vocab_size = len(word_index) + 1
input_length = x_train.shape[1]
embed_dim = embed_dims

rnn_input = Input(shape=(input_length,))

embedding = Embedding(vocab_size,
                              embed_dim,
                              weights=[embedding_matrix],
                              input_length=input_length,
                              name="embedding")(rnn_input)

spatial_dropout_1 = SpatialDropout1D(0.3, name="spatial_dropout")(embedding)

noise = GaussianNoise(0.2, name="noise")(spatial_dropout_1)

bi_gru_1 = Bidirectional(LSTM(RNN_UNITS,
                                           return_sequences=True,
                                           recurrent_regularizer=l2(L2_REG),
                                           kernel_regularizer=l2(L2_REG),
                                           name="bi_gru_1"))(noise)

bi_gru_1 = Dropout(0.3, name="bi_gru_1_dropout")(bi_gru_1)

bi_gru_2 = Bidirectional(LSTM(RNN_UNITS,
                                           return_sequences=True,
                                           recurrent_regularizer=l2(L2_REG),
                                           kernel_regularizer=l2(L2_REG),
                                           name="bi_gru_2"))(bi_gru_1)

bi_gru_2 = Dropout(0.3, name="bi_gru_2_dropout")(bi_gru_2)

attention, weights = Attention(return_attention=True)(bi_gru_2)

drop_1 = Dropout(0.5, name="attention_dropout")(attention)

outputs = Dense(num_classes, activation='softmax', name="output")(drop_1)

global model
model = Model(inputs=rnn_input, outputs=[outputs, weights])

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=LEARN_RATE, clipnorm=CLIP_NORM), metrics=[precision, recall, f1])

model.load_weights('./model_checkpoints/BiLSTMAttention/BiLSTMAttention.hdf5')

val_predictions = model.predict(x=x_val)
# val_classes = lb.inverse_transform(val_predictions)
# confidence = val_predictions.argmax(axis=-1)
#
# print(val_predictions[0])
# print(val_classes[0])
# print(val_predictions[0][confidence[0]])


def predict(text):
    processed = preprocessor.preprocess(text)
    # Number of cells used by this input
    rel_cells = (SEQUENCE_LENGTH - len(processed.split()))
    text = pad_sequences(tokenizer.texts_to_sequences([processed]), maxlen=SEQUENCE_LENGTH)
    prediction, weights = model.predict(x=text)
    pred_class = lb.inverse_transform(prediction)[0]
    confidence = prediction[0][prediction.argmax(axis=-1)][0]
    attn_weights = weights[0][rel_cells:]

    return {
        "processed": processed,
        "classification": pred_class,
        "confidence": float(confidence),
        "attention_weights": attn_weights.tolist(),
    }


@app.route('/status', methods=['GET'])
def status():
    test = json.dumps([{
        'connected': True
    }])
    return test


@app.route('/tweets/train/sample', methods=['GET'])
def tweet_sample():
    sample = sem_eval.sample(n=10)
    test = sample.to_json()
    return test


@app.route('/tweets/process', methods=['POST'])
def process():
    data = request.get_json()

    respon = predict(data['text'])
    return json.dumps([respon])
