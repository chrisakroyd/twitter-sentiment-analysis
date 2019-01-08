import tensorflow as tf
import os
from tqdm import tqdm
from src import config, constants, metrics, models, pipeline, train_utils, util


def train(sess_config, params):
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

        train_tfrecords = util.tf_record_paths(params, training=True)
        val_tfrecords = util.tf_record_paths(params, training=False)
        train_set, train_iter = pipeline.create_pipeline(params, tables, train_tfrecords, training=True)
        _, val_iter = pipeline.create_pipeline(params, tables, val_tfrecords, training=False)

    with tf.Session(config=sess_config) as sess:
        sess.run([tf.tables_initializer(), train_iter.initializer, val_iter.initializer])
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, train_set.output_types, train_set.output_shapes)

        model = models.LSTMAttention(word_matrix, character_matrix, trainable_matrix, 3, params)

        placeholders = iterator.get_next()
        # Features and labels.
        model_inputs = train_utils.inputs_as_tuple(placeholders)
        label_tensor = train_utils.labels_as_tuple(placeholders)[0]
        logits, pred, _ = model(model_inputs, training=True)

        loss_op = model.compute_loss(logits, label_tensor, l2=params.l2)

        train_op = train_utils.construct_train_op(loss_op,
                                                  learn_rate=params.learn_rate,
                                                  warmup_scheme=params.warmup_scheme,
                                                  warmup_steps=params.warmup_steps,
                                                  clip_norm=params.gradient_clip,
                                                  ema_decay=params.ema_decay,
                                                  beta1=params.beta1,
                                                  beta2=params.beta2,
                                                  epsilon=params.epsilon)

        train_outputs = [loss_op, pred, label_tensor, train_op]
        val_outputs = [loss_op, pred, label_tensor]
        sess.run(tf.global_variables_initializer())
        # Saver boilerplate
        writer = tf.summary.FileWriter(log_dir, graph=sess.graph)
        saver = train_utils.get_saver()
        # Initialize the handles for switching.
        train_handle = sess.run(train_iter.string_handle())
        val_handle = sess.run(val_iter.string_handle())

        if os.path.exists(model_dir) and tf.train.latest_checkpoint(model_dir) is not None:
            saver.restore(sess, tf.train.latest_checkpoint(model_dir))

        total_steps = int((params.epochs * meta['num_train'] / params.batch_size))
        global_step = max(sess.run(model.global_step), 1)

        pbar = tqdm(total=total_steps)
        # Set pbar to where we left off.
        pbar.update(global_step)
        train_preds = []

        for _ in range(int(total_steps - global_step)):
            global_step = sess.run(model.global_step) + 1
            loss, pred, label, _ = sess.run(fetches=train_outputs, feed_dict={handle: train_handle})
            train_preds.append((loss, pred, label, ))
            pbar.update()
            # Save at the end of each epoch
            if (global_step % (meta['num_train'] // params.batch_size)) == 0:
                val_preds = []
                for _ in range(meta['num_val']):
                    loss, pred, label = sess.run(fetches=val_outputs,
                                                 feed_dict={
                                                    handle: val_handle,
                                                    model.dropout: 0.0,
                                                    model.attn_dropout: 0.0,
                                                  })
                    val_preds.append((loss, pred, label, ))

                metrics.evaluate_list(train_preds, 'train', writer, global_step)
                val_metrics = metrics.evaluate_list(val_preds, 'val', writer, global_step)

                print('Epoch 1: val_recall:{recall}, val_precision: {prec}, val_f1: {f1}'.format(
                    recall=val_metrics['recall'],
                    prec=val_metrics['precision'],
                    f1=val_metrics['f1']
                ))

                writer.flush()
                filename = os.path.join(model_dir, 'model_{}.ckpt'.format(global_step))
                # Save the model
                saver.save(sess, filename)
                train_preds = []


if __name__ == '__main__':
    defaults = util.namespace_json(path=constants.FilePaths.DEFAULTS)
    train(config.gpu_config(), config.model_config(defaults))
