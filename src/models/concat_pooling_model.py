import tensorflow as tf
from tensorflow.keras.layers import Activation, Bidirectional, CuDNNLSTM, Dense, Dropout, GlobalAveragePooling1D, GlobalMaxPool1D
from src import layers, train_utils, util, constants


class ConcatPoolingModel(tf.keras.Model):
    def __init__(self, embedding_matrix, char_matrix, trainable_matrix, num_classes, params, **kwargs):
        super(ConcatPoolingModel, self).__init__(**kwargs)
        self.global_step = tf.train.get_or_create_global_step()
        self.use_pos_tags = params.use_pos_tags
        self.use_top_k = params.use_top_k

        self.embedding = layers.EmbeddingLayer(embedding_matrix, trainable_matrix, char_matrix,
                                               word_dim=params.embed_dim, char_dim=params.char_dim,
                                               word_dropout=params.word_dropout, char_dropout=params.word_dropout / 2)

        self.rnn_stack = layers.RNNStack(params.rnn_type, params.hidden_units, params.rnn_layers - 1, dropout=params.rnn_dropout,
                                         skip_connection=params.use_rnn_skip_connection, cudnn=params.cudnn,
                                         bidirectional=True, return_sequences=True)

        self.rnn_state = layers.RNNBlock(params.rnn_type, params.hidden_units, dropout=params.rnn_dropout,
                                         skip_connection=params.use_rnn_skip_connection, cudnn=params.cudnn,
                                         bidirectional=True, return_sequences=True, return_state=True)

        self.average_pool = GlobalAveragePooling1D()
        self.max_pool = GlobalMaxPool1D()
        bi_rnn_units = params.hidden_units * 2
        hidden_units = 3 * bi_rnn_units  # forward_rnn + backward_rnn + avg_pool + max_pool

        if self.use_top_k:
            self.top_k_pool = layers.TopKPooling(k=params.top_k)
            hidden_units = hidden_units + (params.top_k * bi_rnn_units)

        self.hidden_stack = layers.HiddenStack(hidden_units, params.hidden_layers, dropout=params.hidden_dropout,
                                               activation='relu', skip_connection=params.use_hidden_skip_connection)

        self.pooled_dropout = Dropout(params.hidden_dropout)
        self.out = Dense(num_classes, name='output')
        self.preds = Activation('softmax')

    def call(self, x, training=None, mask=None):
        words, chars, tags, num_tokens = util.unpack_dict(x, keys=constants.PlaceholderKeys.DEFAULT_INPUTS)
        text_emb = self.embedding([words, chars], training=training)

        if self.use_pos_tags:
            text_emb = tf.concat([text_emb, tags], axis=-1)

        rnn_out = self.rnn_stack(text_emb, training=training)
        rnn_out, last_state = self.rnn_state(rnn_out)

        average_pool = self.average_pool(rnn_out)
        max_pool = self.max_pool(rnn_out)
        pooling_vectors = [average_pool, max_pool, last_state]

        if self.use_top_k:
            top_k_pool = self.top_k_pool(rnn_out)
            pooling_vectors = pooling_vectors + [top_k_pool]

        pooled = tf.concat(pooling_vectors, axis=-1)
        pooled = self.pooled_dropout(pooled, training=training)
        pooled = self.hidden_stack(pooled, training=training)

        logits = self.out(pooled)
        preds = self.preds(logits)

        return logits, preds

    def compute_loss(self, logits, labels, l2=None):
        loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels)
        loss = tf.reduce_mean(loss)

        if l2 is not None and l2 > 0.0:
            variables = tf.trainable_variables()
            variables = [v for v in variables if 'bias' not in v.name and 'scale' not in v.name]
            l2_loss = train_utils.l2_ops(l2, variables=variables)
            loss = loss + l2_loss

        return loss
