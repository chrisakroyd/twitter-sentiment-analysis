from keras.layers import Dense, Embedding, Bidirectional, Dropout, BatchNormalization, SpatialDropout1D, CuDNNGRU
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop

from .layers.Attention import HierarchicalAttention as Attention
from .metrics import f1

# HPARAMs
BATCH_SIZE = 64
EPOCHS = 50
LEARN_RATE = 0.001
CLIP_NORM = 1.0
NUM_CLASSES = 12


class CUDNNBiRNNAttention:
    def __init__(self, num_classes=5):
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.LEARN_RATE = LEARN_RATE
        self.num_classes = num_classes
        self.checkpoint_path = './model_checkpoints/CUDNN_RNN.hdf5'

    def create_model(self, vocab_size, embedding_matrix, input_length=5000, embed_dim=100):
        model = Sequential()

        model.add(Embedding(vocab_size, embed_dim, weights=[embedding_matrix], input_length=input_length))
        model.add(SpatialDropout1D(0.2))
        model.add(BatchNormalization())

        model.add(Bidirectional(CuDNNGRU(150, return_sequences=True)))
        model.add(Bidirectional(CuDNNGRU(150, return_sequences=True)))

        model.add(Attention())
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))

        model.summary()

        model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=self.LEARN_RATE, clipnorm=CLIP_NORM), metrics=['accuracy', f1])

        return model
