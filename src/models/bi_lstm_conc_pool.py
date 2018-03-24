from keras.layers import Input, Dense, Embedding, Bidirectional, SpatialDropout1D, \
    GaussianNoise, CuDNNLSTM, concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D, Dropout, BatchNormalization, Lambda
from keras.models import Model
from keras.optimizers import Nadam, RMSprop
from src.metrics import f1
from ..layers.Attention import FeedForwardAttention as Attention

from keras.backend import squeeze, sum

import tensorflow as tf

# HPARAMs
BATCH_SIZE = 128
EPOCHS = 50
LEARN_RATE = 0.0005
NUM_CLASSES = 12

top_k = 10


def _top_k(x):
    x = tf.transpose(x, [0, 2, 1])
    k_max = tf.nn.top_k(x, k=top_k)
    return tf.reshape(k_max[0], (-1, 2 * 80 * top_k))


class BiLSTMConcPool:
    def __init__(self, num_classes=NUM_CLASSES):
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.LEARN_RATE = LEARN_RATE
        self.num_classes = num_classes

    def create_model(self, vocab_size, embedding_matrix, afinn_matrix, input_length=5000, embed_dim=200):
        input = Input(shape=(input_length, ))

        embedding = Embedding(vocab_size, embed_dim, weights=[embedding_matrix], input_length=input_length)(input)
        affin_embedding = Embedding(vocab_size, 1, weights=[afinn_matrix], input_length=input_length)(input)

        spatial_dropout_1 = SpatialDropout1D(0.2)(embedding)

        noise = GaussianNoise(0.2)(spatial_dropout_1)

        batch_norm = BatchNormalization()(noise)

        bi_gru_1 = Bidirectional(CuDNNLSTM(80, return_sequences=True, recurrent_initializer='he_normal'))(batch_norm)

        bi_gru_1 = SpatialDropout1D(0.25)(bi_gru_1)

        bi_gru_2, forward_h, backward_h, forward_c, backward_c = Bidirectional(CuDNNLSTM(80,
                                                                                         return_sequences=True,
                                                                                         return_state=True,
                                                                                         recurrent_initializer='he_normal'))(bi_gru_1)

        bi_gru_2 = SpatialDropout1D(0.25)(bi_gru_2)

        last_state = concatenate([forward_h, backward_h], name='last_state')
        avg_pool = GlobalAveragePooling1D()(bi_gru_2)
        k_max = Lambda(_top_k)(bi_gru_2)
        affin_vec = Lambda(lambda x: sum(x, axis=-1))(affin_embedding)

        conc = concatenate([last_state, k_max, avg_pool, affin_vec], name='conc_pool')

        drop_1 = Dropout(0.5)(conc)

        outputs = Dense(self.num_classes, activation='softmax')(drop_1)

        model = Model(inputs=input, outputs=outputs)

        return model

    def build(self, vocab_size, embedding_matrix, afinn_matrix, input_length=5000, embed_dim=200, summary=True):
        # model = self.create_model(vocab_size, embedding_matrix, input_length, embed_dim)
        model = self.create_model(vocab_size, embedding_matrix, afinn_matrix, input_length, embed_dim)

        model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(lr=self.LEARN_RATE),
                      metrics=['accuracy', f1])

        if summary:
            model.summary()

        return model
