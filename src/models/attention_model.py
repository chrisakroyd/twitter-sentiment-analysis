import tensorflow as tf
from tensorflow.keras.layers import Activation, Bidirectional, CuDNNLSTM, Dense, Dropout, Embedding
from src import layers, train_utils, util, constants


class AttentionModel(tf.keras.Model):
    def __init__(self, embedding_matrix, char_matrix, trainable_matrix, num_classes, params, **kwargs):
        super(AttentionModel, self).__init__(**kwargs)
        self.global_step = tf.train.get_or_create_global_step()
        self.use_pos_tags = params.use_pos_tags

        self.embedding = layers.EmbeddingLayer(embedding_matrix, trainable_matrix, char_matrix,
                                               word_dim=params.embed_dim, char_dim=params.char_dim,
                                               word_dropout=params.word_dropout, char_dropout=params.word_dropout / 2)

        self.rnn_stack = layers.RNNStack(params.rnn_type, params.hidden_units, params.rnn_layers, dropout=params.rnn_dropout,
                                         skip_connection=params.use_rnn_skip_connection, cudnn=params.cudnn,
                                         bidirectional=True, return_sequences=True)

        self.attention = layers.Attention(return_attention=True)

        self.attention_dropout = Dropout(params.attn_dropout, name='attention_dropout')

        self.out = Dense(num_classes, name='output')
        self.preds = Activation('softmax')

    def call(self, x, training=None, mask=None):
        words, chars, tags, num_tokens = util.unpack_dict(x, keys=constants.PlaceholderKeys.DEFAULT_INPUTS)
        attn_mask = layers.create_mask(num_tokens, maxlen=tf.reduce_max(num_tokens))
        text_emb = self.embedding([words, chars], training=training)

        if self.use_pos_tags:
            text_emb = tf.concat([text_emb, tags], axis=-1)

        rnn_out = self.rnn_stack(text_emb, training=training)
        attn_out, attn_weights = self.attention(rnn_out, mask=attn_mask)
        attn_out = self.attention_dropout(attn_out, training=training)

        logits = self.out(attn_out)
        preds = self.preds(logits)

        return logits, preds, attn_weights

    def compute_loss(self, logits, labels, l2=None):
        loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels)
        loss = tf.reduce_mean(loss)

        if l2 is not None and l2 > 0.0:
            variables = tf.trainable_variables()
            variables = [v for v in variables if 'bias' not in v.name and 'scale' not in v.name]
            l2_loss = train_utils.l2_ops(l2, variables=variables)
            loss = loss + l2_loss

        return loss
