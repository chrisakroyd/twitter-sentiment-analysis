from keras.layers import Conv1D, Dense, Embedding, GlobalMaxPool1D, MaxPooling1D, concatenate, Input, Dropout, BatchNormalization, SpatialDropout1D
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from src.metrics import f1

# HPARAMs
BATCH_SIZE = 64
EPOCHS = 50
LEARN_RATE = 0.0001
NUM_CLASSES = 12


class CNNModel:
    def __init__(self, num_classes=5):
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.LEARN_RATE = LEARN_RATE
        self.num_classes = num_classes

    def create_model(self, vocab_size, embedding_matrix, input_length=5000, embed_dim=100):

        input_text = Input(shape=(input_length,), dtype='int32')

        embed = Embedding(vocab_size, embed_dim, weights=[embedding_matrix], input_length=input_length)(input_text)
        embed_drop = SpatialDropout1D(0.2)(embed)
        batch_1 = BatchNormalization()(embed_drop)

        feat_maps_1 = Conv1D(100, kernel_size=3, padding="same", activation="relu")(batch_1)
        pool_vecs_1 = MaxPooling1D(pool_length=2)(feat_maps_1)
        pool_vecs_1 = GlobalMaxPool1D()(pool_vecs_1)

        feat_maps_2 = Conv1D(100, kernel_size=4, padding="same", activation="relu")(batch_1)
        pool_vecs_2 = MaxPooling1D(pool_size=2)(feat_maps_2)
        pool_vecs_2 = GlobalMaxPool1D()(pool_vecs_2)

        feat_maps_3 = Conv1D(100, kernel_size=5, padding="same", activation="relu")(batch_1)
        pool_vecs_3 = MaxPooling1D(pool_size=2)(feat_maps_3)
        pool_vecs_3 = GlobalMaxPool1D()(pool_vecs_3)

        concat_1 = concatenate([pool_vecs_1, pool_vecs_2, pool_vecs_3])

        drop_1 = Dropout(0.5)(concat_1)

        output = Dense(self.num_classes, activation='softmax', activity_regularizer=l2(0.0001))(drop_1)

        model = Model(input=input_text, output=output)

        model.summary()

        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.LEARN_RATE), metrics=['acc', f1])

        return model

