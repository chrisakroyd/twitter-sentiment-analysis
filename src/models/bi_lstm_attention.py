from keras.layers import Dense, Embedding, Bidirectional, Dropout, BatchNormalization, SpatialDropout1D, CuDNNLSTM, GaussianNoise
from keras.models import Sequential
from keras.regularizers import l2
from keras.optimizers import RMSprop, Nadam, Adam

from ..layers.Attention import FeedForwardAttention as Attention
from src.metrics import f1

# HPARAMs
BATCH_SIZE = 128
EPOCHS = 50
LEARN_RATE = 0.001
CLIP_NORM = 1.0
NUM_CLASSES = 12


class BiLSTMAttention:
    def __init__(self, num_classes=5):
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.LEARN_RATE = LEARN_RATE
        self.num_classes = num_classes

    def create_model(self, vocab_size, embedding_matrix, input_length=5000, embed_dim=200):
        model = Sequential()

        model.add(Embedding(vocab_size, embed_dim, weights=[embedding_matrix], input_length=input_length))
        model.add(SpatialDropout1D(0.3))
        model.add(GaussianNoise(0.2))
        model.add(BatchNormalization())

        model.add(Bidirectional(CuDNNLSTM(150,
                                          return_sequences=True,
                                          recurrent_regularizer=l2(0.0001),
                                          kernel_regularizer=l2(0.0001))))
        model.add(Dropout(0.3))

        model.add(Bidirectional(CuDNNLSTM(150,
                                          return_sequences=True,
                                          recurrent_regularizer=l2(0.0001),
                                          kernel_regularizer=l2(0.0001))))
        model.add(Dropout(0.3))

        model.add(Attention())
        model.add(Dropout(0.5))

        model.add(Dense(self.num_classes, activation='softmax'))

        return model

    def build(self, vocab_size, embedding_matrix, afinn_matrix, input_length=5000, embed_dim=200, summary=True):
        model = self.create_model(vocab_size, embedding_matrix, input_length, embed_dim)

        model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=self.LEARN_RATE, clipnorm=CLIP_NORM),
        # model.compile(loss='categorical_crossentropy', optimizer=Nadam(lr=self.LEARN_RATE, clipnorm=CLIP_NORM),
        # model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.LEARN_RATE, clipnorm=CLIP_NORM),
                      metrics=['accuracy', f1])

        if summary:
            model.summary()

        return model