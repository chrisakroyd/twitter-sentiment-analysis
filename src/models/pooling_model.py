import tensorflow as tf
from tensorflow.keras.layers import Activation, Bidirectional, CuDNNLSTM, Dense, Dropout, GlobalAveragePooling1D, GlobalMaxPool1D
from src import layers, train_utils


class PoolingModel(tf.keras.Model):
    def __init__(self, embedding_matrix, char_matrix, trainable_matrix, num_classes, params, **kwargs):
        super(PoolingModel, self).__init__(**kwargs)
        self.global_step = tf.train.get_or_create_global_step()
        self.dropout = tf.placeholder_with_default(params.dropout, (), name='dropout')
        self.attn_dropout = tf.placeholder_with_default(params.attn_dropout, (), name='attn_dropout')
        self.use_pos_tags = params.use_pos_tags
        self.use_top_k = params.use_top_k

        self.embedding = layers.EmbeddingLayer(embedding_matrix, trainable_matrix, char_matrix,
                                               word_dim=params.embed_dim, char_dim=params.char_dim,
                                               word_dropout=self.dropout, char_dropout=self.dropout / 2)

        self.rnn_1 = layers.RNNBlock(params.rnn_type, params.hidden_units, dropout=self.dropout,
                                     skip_connection=params.use_rnn_skip_connection, cudnn=params.cudnn,
                                     bidirectional=True, return_sequences=True, name='bi_gru_1')

        self.rnn_2 = layers.RNNBlock(params.rnn_type, params.hidden_units, dropout=self.dropout,
                                     skip_connection=params.use_rnn_skip_connection, cudnn=params.cudnn,
                                     bidirectional=True, return_sequences=True, name='bi_gru_1')

        self.average_pool = GlobalAveragePooling1D()
        self.max_pool = GlobalMaxPool1D()
        self.top_k_pool = layers.TopKPooling(k=params.top_k)

        self.drop_3 = Dropout(self.attn_dropout, name='attention_dropout')

        self.out = Dense(num_classes, name='output')
        self.preds = Activation('softmax')

    def call(self, x, training=None, mask=None):
        words, chars, tags, num_tokens = x
        text_emb = self.embedding([words, chars], training=training)

        if self.use_pos_tags:
            text_emb = tf.concat([text_emb, tags], axis=-1)

        rnn_1_out = self.rnn_1(text_emb, training=training)
        rnn_2_out = self.rnn_2(rnn_1_out, training=training)

        average_pool = self.average_pool(rnn_2_out)
        max_pool = self.max_pool(rnn_2_out)
        pooling_vectors = [average_pool, max_pool]

        if self.use_top_k:
            top_k_pool = self.top_k_pool(rnn_2_out)
            pooling_vectors = pooling_vectors + [top_k_pool]

        pooled = tf.concat(pooling_vectors, axis=-1)
        logits = self.out(pooled)
        preds = self.preds(logits)

        return logits, preds

    def compute_loss(self, logits, labels, l2=None):
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
        loss = tf.reduce_mean(loss)

        if l2 is not None and l2 > 0.0:
            variables = tf.trainable_variables()
            variables = [v for v in variables if 'bias' not in v.name and 'scale' not in v.name]
            l2_loss = train_utils.l2_ops(l2, variables=variables)
            loss = loss + l2_loss

        return loss
