from keras.layers import Input, Dense, Bidirectional, Dropout, CuDNNLSTM
from keras.regularizers import l2
from keras.optimizers import RMSprop
from keras.models import Model
from ..layers.Attention import FeedForwardAttention as Attention
from src.metrics import f1, precision, recall
from src.models.TextModel import TextModel

# HPARAMs
BATCH_SIZE = 128
EPOCHS = 50
LEARN_RATE = 0.001
CLIP_NORM = 5.0
NUM_CLASSES = 12
RNN_UNITS = 150
L2_REG = 0.0001


class BiLSTMAttention(TextModel):
    def __init__(self, num_classes=5):
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.LEARN_RATE = LEARN_RATE
        self.num_classes = num_classes

    def create_model(self, vocab_size, embedding_matrix, input_length=5000, embed_dim=200):
        rnn_input = Input(shape=(input_length,))
        embedding = self.embedding_layers(rnn_input, vocab_size, embedding_matrix,
                                          dropout=0.3, noise=0.2,
                                          input_length=input_length, embed_dim=embed_dim)

        bi_gru_1 = Bidirectional(CuDNNLSTM(RNN_UNITS,
                                           return_sequences=True,
                                           recurrent_regularizer=l2(L2_REG),
                                           kernel_regularizer=l2(L2_REG)))(embedding)

        bi_gru_1 = Dropout(0.3)(bi_gru_1)

        bi_gru_2 = Bidirectional(CuDNNLSTM(RNN_UNITS,
                                           return_sequences=True,
                                           recurrent_regularizer=l2(L2_REG),
                                           kernel_regularizer=l2(L2_REG)))(bi_gru_1)

        bi_gru_2 = Dropout(0.3)(bi_gru_2)

        attention = Attention()(bi_gru_2)

        drop_1 = Dropout(0.5)(attention)

        outputs = Dense(self.num_classes, activation='softmax')(drop_1)

        model = Model(inputs=rnn_input, outputs=outputs)

        return model

    def build(self, vocab_size, embedding_matrix, afinn_matrix, input_length=5000, embed_dim=200, summary=True):
        model = self.create_model(vocab_size, embedding_matrix, input_length, embed_dim)

        model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=self.LEARN_RATE, clipnorm=CLIP_NORM),
        # model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=self.LEARN_RATE),
                      metrics=[precision, recall, f1])

        if summary:
            model.summary()

        return model
