import tensorflow as tf
from tqdm import tqdm
from src import config, constants, metrics, models, pipeline, train_utils, util


def test(sess_config, params):
    _, out_dir, model_dir, log_dir = util.train_paths(params)
    word_index_path, _, char_index_path = util.index_paths(params)
    embedding_paths = util.embedding_paths(params)
    meta_path = util.meta_path(params)
    util.make_dirs([out_dir, model_dir, log_dir])

    vocabs = util.load_vocab_files(paths=(word_index_path, char_index_path))
    word_matrix, trainable_matrix, character_matrix = util.load_numpy_files(paths=embedding_paths)
    meta = util.load_json(meta_path)

    with tf.device('/cpu:0'):
        tables = pipeline.create_lookup_tables(vocabs)
        val_tfrecords = util.tf_record_paths(params, training=False)
        val_set, iterator = pipeline.create_pipeline(params, tables, val_tfrecords, training=False)

    with tf.Session(config=sess_config) as sess:
        sess.run(iterator.initializer)
        sess.run(tf.tables_initializer())

        model = models.LSTMAttention(word_matrix, character_matrix, trainable_matrix, meta['num_classes'], params)

        placeholders = iterator.get_next()
        # Features and labels.
        model_inputs = train_utils.inputs_as_tuple(placeholders)
        label_tensor = train_utils.labels_as_tuple(placeholders)[0]
        logits, prediction, _ = model(model_inputs, training=False)

        sess.run(tf.global_variables_initializer())

        # Restore the moving average version of the learned variables for eval.
        saver = train_utils.get_saver(ema_decay=params.ema_decay, ema_vars_only=True)
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        preds = []
        # +1 for uneven batch values, +1 for the range.
        for _ in tqdm(range(1, (meta['num_val'] // params.batch_size + 1) + 1)):
            pred, label = sess.run([prediction, label_tensor],
                                   feed_dict={
                                       model.dropout: 0.0,
                                       model.attn_dropout: 0.0,
                                   })
            preds.append((0.0, pred, label,))

        # Evaluate the predictions and reset the train result list for next eval period.
        val_metrics = metrics.evaluate_list(preds)
        print('EMA results: val_recall:{recall}, val_precision: {prec}, val_f1: {f1}'.format(
            recall=val_metrics['recall'],
            prec=val_metrics['precision'],
            f1=val_metrics['f1']
        ))


if __name__ == '__main__':
    defaults = util.namespace_json(path=constants.FilePaths.DEFAULTS)
    test(config.gpu_config(), config.model_config(defaults).FLAGS)