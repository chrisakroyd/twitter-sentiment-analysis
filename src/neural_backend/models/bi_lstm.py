from keras.layers import Dense, Embedding, LSTM, Bidirectional, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from src.metrics import f1
from src.models.TextModel import TextModel

# HPARAMs
BATCH_SIZE = 64
EPOCHS = 50
LEARN_RATE = 0.0001
NUM_CLASSES = 12


class BiLSTMModel(TextModel):
    def __init__(self, num_classes=5):
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.LEARN_RATE = LEARN_RATE
        self.num_classes = num_classes

    def create_model(self, vocab_size, embedding_matrix, input_length=5000, embed_dim=100):
        model = Sequential()

        model.add(Embedding(vocab_size, embed_dim, weights=[embedding_matrix], input_length=input_length, dropout=0.2))
        model.add(BatchNormalization())

        model.add(Bidirectional(LSTM(75, return_sequences=True)))
        model.add(Bidirectional(LSTM(75)))
        model.add(Dense(self.num_classes, activation='softmax'))

        model.summary()

        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.LEARN_RATE), metrics=['accuracy', f1])

        return model
