import tensorflow as tf
import os
from tqdm import tqdm
from src import config, constants, metrics, models, pipeline, train_utils, util


def train(sess_config, params):
    model_dir, log_dir = util.save_paths(params)
    word_index_path, _, char_index_path, pos_index_path = util.index_paths(params)
    embedding_paths = util.embedding_paths(params)
    meta_path = util.meta_path(params)
    util.make_dirs([model_dir, log_dir])

    vocabs = util.load_vocab_files(paths=(word_index_path, char_index_path, pos_index_path))
    word_matrix, trainable_matrix, character_matrix = util.load_numpy_files(paths=embedding_paths)
    meta = util.load_json(meta_path)
    num_classes = meta['num_classes']
    num_tags = meta['num_tags']

    with tf.device('/cpu:0'):
        tables = pipeline.create_lookup_tables(vocabs)

        train_tfrecords, val_tfrecords = util.tf_record_paths(params)
        train_set, train_iter = pipeline.create_pipeline(params, tables, train_tfrecords, num_classes, num_tags, training=True)
        _, val_iter = pipeline.create_pipeline(params, tables, val_tfrecords, num_classes, num_tags, training=False)

    with tf.Session(config=sess_config) as sess:
        sess.run([tf.tables_initializer(), train_iter.initializer, val_iter.initializer])
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, train_set.output_types, train_set.output_shapes)

        if params.model_type == constants.ModelTypes.ATTENTION:
            model = models.AttentionModel(word_matrix, character_matrix, trainable_matrix, num_classes, params)
        elif params.model_type == constants.ModelTypes.CONC_POOL:
            model = models.ConcatPoolingModel(word_matrix, character_matrix, trainable_matrix, num_classes, params)
        elif params.model_type == constants.ModelTypes.POOL:
            model = models.PoolingModel(word_matrix, character_matrix, trainable_matrix, num_classes, params)
        else:
            raise ValueError(constants.ErrorMessages.INVALID_MODEL_TYPE.format(model_type=params.model_type))

        placeholders = iterator.get_next()
        training_flag = tf.placeholder_with_default(True, (), name='training_flag')
        # Features and labels.
        model_inputs = train_utils.inputs_as_tuple(placeholders)
        label_tensor = train_utils.labels_as_tuple(placeholders)[0]

        if params.model_type == constants.ModelTypes.ATTENTION:
            logits, prediction, _ = model(model_inputs, training=training_flag)
        else:
            logits, prediction = model(model_inputs, training=training_flag)

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

        recall = metrics.recall(label_tensor, prediction)
        precision = metrics.precision(label_tensor, prediction)
        f1 = metrics.harmonic_mean(precision, recall)

        train_outputs = [recall, precision, f1, train_op]
        val_outputs = [recall, precision, f1]
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
            recall, precision, f1, _ = sess.run(fetches=train_outputs, feed_dict={handle: train_handle})
            train_preds.append((recall, precision, f1, ))
            pbar.update()
            # Save at the end of each epoch
            if (global_step % (meta['num_train'] // params.batch_size)) == 0 or global_step == total_steps:
                val_preds = []
                for _ in tqdm(range(meta['num_val'] // params.batch_size)):
                    recall, precision, f1 = sess.run(fetches=val_outputs,
                                                     feed_dict={
                                                        handle: val_handle,
                                                        training_flag: False,
                                                      })
                    val_preds.append((recall, precision, f1, ))

                metrics.evaluate_list(train_preds, 'train', writer, global_step, subsample_ratio=0.1)
                val_metrics = metrics.evaluate_list(val_preds, 'val', writer, global_step)

                print('Epoch {num}: val_recall:{recall}, val_precision: {prec}, val_f1: {f1}'.format(
                    num=int(global_step / (meta['num_train'] // params.batch_size)),
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
