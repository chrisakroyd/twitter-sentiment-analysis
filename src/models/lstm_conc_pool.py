from keras.layers import Input, Dense, Embedding, Bidirectional, SpatialDropout1D, CuDNNLSTM, concatenate,\
    GlobalAveragePooling1D, Dropout, Lambda
from keras.models import Model
from keras.optimizers import Nadam, RMSprop, Adam
from src.metrics import f1
from keras.backend import sum, epsilon
from keras.regularizers import l2
from src.models.TextModel import ConcPoolModel

import tensorflow as tf

# HPARAMs
BATCH_SIZE = 128
EPOCHS = 50
LEARN_RATE = 0.001
CLIP_NORM = 5.0
NUM_CLASSES = 12
RNN_UNITS = 150
L2_REG = 0.0001

top_k = 10


def _top_k(x):
    x = tf.transpose(x, [0, 2, 1])
    k_max = tf.nn.top_k(x, k=top_k)
    return tf.reshape(k_max[0], (-1, 2 * RNN_UNITS * top_k))


class BiLSTMConcPool(ConcPoolModel):
    def __init__(self, num_classes=NUM_CLASSES):
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.LEARN_RATE = LEARN_RATE
        self.num_classes = num_classes

    def create_model(self, vocab_size, embedding_matrix, afinn_matrix, input_length=5000, embed_dim=200):
        rnn_input = Input(shape=(input_length,))
        embedding = self.embedding_layers(rnn_input, vocab_size, embedding_matrix,
                                          dropout=0.3, noise=0.2,
                                          input_length=input_length, embed_dim=embed_dim)

        affin_embedding = Embedding(vocab_size, 1, weights=[afinn_matrix], input_length=input_length)(rnn_input)

        bi_gru_1 = Bidirectional(CuDNNLSTM(RNN_UNITS,
                                           return_sequences=True,
                                           recurrent_regularizer=l2(L2_REG),
                                           kernel_regularizer=l2(L2_REG),
                                           recurrent_initializer='he_normal'))(embedding)

        bi_gru_1 = SpatialDropout1D(0.2)(bi_gru_1)

        bi_gru_2, forward_h, backward_h, forward_c, backward_c = Bidirectional(CuDNNLSTM(RNN_UNITS,
                                                                                         return_sequences=True,
                                                                                         return_state=True,
                                                                                         recurrent_regularizer=l2(L2_REG),
                                                                                         kernel_regularizer=l2(L2_REG),
                                                                                         recurrent_initializer='he_normal'))(bi_gru_1)

        bi_gru_2 = SpatialDropout1D(0.2)(bi_gru_2)

        bi_gru_2 = concatenate([bi_gru_2, bi_gru_1, embedding])

        last_state = self.concat_state(forward_h, backward_h)

        avg_pool = GlobalAveragePooling1D()(bi_gru_2)
        k_max = Lambda(_top_k)(bi_gru_2)
        # affin_vec = Lambda(lambda x: tf.divide(sum(x, axis=1) + epsilon(), input_length))(affin_embedding)
        affin_vec = Lambda(lambda x: sum(x, axis=1))(affin_embedding)

        # conc = concatenate([last_state, k_max, avg_pool, affin_vec], name='conc_pool')
        conc = concatenate([last_state, k_max, avg_pool], name='conc_pool')

        drop_1 = Dropout(0.65)(conc)
        outputs = Dense(self.num_classes, activation='softmax')(drop_1)

        model = Model(inputs=rnn_input, outputs=outputs)

        return model

    def build(self, vocab_size, embedding_matrix, afinn_matrix, input_length=5000, embed_dim=200, summary=True):
        # model = self.create_model(vocab_size, embedding_matrix, input_length, embed_dim)
        model = self.create_model(vocab_size, embedding_matrix, afinn_matrix, input_length, embed_dim)

        model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(lr=self.LEARN_RATE, clipnorm=CLIP_NORM),
                      # optimizer=Adam(lr=self.LEARN_RATE, clipnorm=CLIP_NORM),
                      metrics=['accuracy', f1])

        if summary:
            model.summary()

        return model
