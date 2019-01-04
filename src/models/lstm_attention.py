import tensorflow as tf
from tensorflow.keras.layers import Activation, Bidirectional, CuDNNLSTM, Dense, Dropout

from src import layers, models, train_utils


class LSTMAttention(tf.keras.models.Model):
    def __init__(self, embedding_matrix, char_matrix, trainable_matrix, num_classes, params, **kwargs):
        super(LSTMAttention, self).__init__(**kwargs)
        self.global_step = tf.train.get_or_create_global_step()
        self.dropout = tf.placeholder_with_default(params.dropout, (), name='dropout')
        self.attn_dropout = tf.placeholder_with_default(params.attn_dropout, (), name='attn_dropout')

        self.embedding = models.EmbeddingLayer(embedding_matrix, trainable_matrix, char_matrix,
                                               word_dim=params.embed_dim, char_dim=params.char_dim,
                                               word_dropout=self.dropout, char_dropout=self.dropout / 2)

        self.rnn_1 = Bidirectional(CuDNNLSTM(params.hidden_units, return_sequences=True, name='bi_gru_1'))

        self.drop_1 = Dropout(self.dropout, name='bi_gru_1_dropout')

        self.rnn_2 = Bidirectional(CuDNNLSTM(params.hidden_units, return_sequences=True, name='bi_gru_2'))

        self.drop_2 = Dropout(self.dropout, name='bi_gru_2_dropout')

        self.attention = layers.Attention()

        self.drop_3 = Dropout(self.attn_dropout, name='attention_dropout')

        self.out = Dense(num_classes, name='output')
        self.preds = Activation('softmax')

    def call(self, x, training=None, mask=None):
        words, chars = x

        text_emb = self.embedding([words, chars], training=training)

        rnn_1_out = self.rnn_1(text_emb)
        rnn_1_out = self.drop_1(rnn_1_out, training=training)

        rnn_2_out = self.rnn_2(rnn_1_out)
        rnn_2_out = self.drop_2(rnn_2_out)

        attn_out = self.attention(rnn_2_out)
        attn_out = self.drop_3(attn_out)

        logits = self.out(attn_out)
        preds = self.preds(logits)

        return logits, preds, attn_out

    def compute_loss(self, start_logits, start_labels, l2=None):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=start_logits, labels=start_labels)

        loss = tf.reduce_mean(loss)

        if l2 is not None and l2 > 0.0:
            l2_loss = train_utils.l2_ops(l2)
            loss = loss + l2_loss

        return loss
