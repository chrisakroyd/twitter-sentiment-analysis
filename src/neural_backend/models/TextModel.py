from keras.layers import concatenate, GlobalMaxPooling1D, GlobalAveragePooling1D, Embedding, SpatialDropout1D, \
    GaussianNoise


class TextModel:
    def embedding_layers(self, tensor, vocab_size, embedding_matrix, dropout=0.5, noise=0.2, input_length=5000,
                         embed_dim=200, embeddings_regularizer=None):
        embedding = Embedding(vocab_size,
                              embed_dim,
                              weights=[embedding_matrix],
                              embeddings_regularizer=embeddings_regularizer,
                              input_length=input_length,
                              name="embedding")(tensor)

        spatial_dropout_1 = SpatialDropout1D(dropout, name="spatial_dropout")(embedding)

        noise = GaussianNoise(noise, name="noise")(spatial_dropout_1)

        return noise

    def create_model(self, vocab_size, embedding_matrix, input_length=5000, embed_dim=200):
        pass

    def build(self, vocab_size, embedding_matrix, input_length=5000, embed_dim=200, summary=True):
        pass


class ConcPoolModel(TextModel):
    def concat_state(self, state_forward, state_back):
        return concatenate([state_forward, state_back], name='last_state')

    def bi_concatenate_pool(self, rnn, state_forward, state_back):
        return self.concatenate_pool(rnn, self.concat_state(state_forward, state_back))

    def concatenate_pool(self, rnn, state):
        avg_pool = GlobalAveragePooling1D()(rnn)
        max_pool = GlobalMaxPooling1D()(rnn)

        concatenate_pool = concatenate([state, max_pool, avg_pool], name='concatenate_pool')
        return concatenate_pool


class MaxAvgPoolModel(TextModel):
    def max_avg_pool(self, rnn):
        avg_pool = GlobalAveragePooling1D()(rnn)
        max_pool = GlobalMaxPooling1D()(rnn)

        concatenate_pool = concatenate([max_pool, avg_pool], name='max_avg_pool')
        return concatenate_pool
