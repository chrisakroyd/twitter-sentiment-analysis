import os
import random
import numpy as np
import tensorflow as tf
# from src import config, constants, demo_utils, models, pipeline, preprocessing as prepro, train_utils, util
from src import config, constants, demo_utils, models, pipeline, train_utils, util, tokenizer as toke
from src.preprocessor import TextPreProcessor

API_VERSION = 1
BAD_REQUEST_CODE = 400


def demo(sess_config, params):
    # Although bad practice, I don't want to force people to install unnecessary dependencies to run this repo.
    from flask import Flask, json, request, send_from_directory

    # TODO This is a mess and shouldn't be here but is neccessary for demo_ui development.
    # Comes from https://gist.github.com/blixt/54d0a8bf9f64ce2ec6b8
    def add_cors_headers(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        if request.method == 'OPTIONS':
            response.headers['Access-Control-Allow-Methods'] = 'DELETE, GET, POST, PUT'
            headers = request.headers.get('Access-Control-Request-Headers')
            if headers:
                response.headers['Access-Control-Allow-Headers'] = headers
        return response

    app = Flask(__name__, static_folder=params.dist_dir)
    app.after_request(add_cors_headers)

    model_dir, _ = util.save_paths(params)
    word_index_path, _, char_index_path = util.index_paths(params)
    examples_path = util.examples_path(params)
    embedding_paths = util.embedding_paths(params)
    meta_path = util.meta_path(params)
    classes_path = util.classes_path(params)

    json_paths = (word_index_path, char_index_path, examples_path, meta_path, classes_path, )
    word_index, char_index, examples, meta, classes = util.load_multiple_jsons(paths=json_paths)
    reverse_classes = {value: key for key, value in classes.items()}

    preprocessor = TextPreProcessor()
    tokenizer = toke.Tokenizer(lower=False,
                               oov_token=params.oov_token,
                               word_index=word_index,
                               char_index=char_index,
                               trainable_words=params.trainable_words,
                               filters=None)

    vocabs = util.load_vocab_files(paths=(word_index_path, char_index_path))
    word_matrix, trainable_matrix, character_matrix = util.load_numpy_files(paths=embedding_paths)
    tables = pipeline.create_lookup_tables(vocabs)
    # Keep sess alive as long as the server is live, probably not best practice but it works @TODO Improve this.
    sess = tf.Session(config=sess_config)
    sess.run(tf.tables_initializer())
    # Initialise the model, pipelines + placeholders.
    model = models.LSTMAttention(word_matrix, character_matrix, trainable_matrix, meta['num_classes'], params)
    pipeline_placeholders = pipeline.create_placeholders()
    demo_dataset, demo_iter = pipeline.create_demo_pipeline(params, tables, pipeline_placeholders)
    demo_placeholders = demo_iter.get_next()
    demo_inputs = train_utils.inputs_as_tuple(demo_placeholders)
    logits, prediction, attn_weights = model(demo_inputs)

    demo_outputs = [logits, prediction, attn_weights]
    sess.run(tf.global_variables_initializer())

    saver = train_utils.get_saver(ema_decay=params.ema_decay, ema_vars_only=True)
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))

    @app.route('/api/v{0}/model/predict'.format(API_VERSION), methods=['POST'])
    def predict():
        data = request.get_json()

        if 'text' not in data:
            return json.dumps(demo_utils.get_error_response(constants.ErrorMessages.NO_TEXT,
                                                            data, error_code=1)), BAD_REQUEST_CODE

        text = preprocessor.preprocess(data['text'])
        tokens, modified_tokens = tokenizer.tokenize(text)

        if len(data['text']) <= 0 or len(tokens) <= 0:
            return json.dumps(demo_utils.get_error_response(constants.ErrorMessages.INVALID_TEXT,
                                                            data, error_code=2)), BAD_REQUEST_CODE

        return json.dumps(process(tokens, data))

    @app.route('/api/v{0}/model/predictTokens'.format(API_VERSION), methods=['POST'])
    def predict_tokens():
        data = request.get_json()

        if 'tokens' not in data:
            return json.dumps(demo_utils.get_error_response(constants.ErrorMessages.NO_TOKENS,
                                                            data, error_code=3)), BAD_REQUEST_CODE

        tokens = data['tokens']

        if len(tokens) <= 0:
            return json.dumps(demo_utils.get_error_response(constants.ErrorMessages.INVALID_TOKENS,
                                                            data, error_code=4)), BAD_REQUEST_CODE

        return json.dumps(process(tokens, data))

    def process(tokens, data):
        sess.run(demo_iter.initializer, feed_dict={
            'tokens:0': np.array([tokens], dtype=np.str),
            'num_tokens:0': np.array([len(tokens)], dtype=np.int32),
        })
        _, probs, attn_out = sess.run(fetches=demo_outputs)
        preds = [reverse_classes[np.argmax(prob)] for prob in probs.tolist()]
        probs = probs.tolist()
        attn_out = attn_out.tolist()
        return demo_utils.get_predict_response(tokens, probs, preds, attn_out, data)

    @app.route('/api/v{0}/examples'.format(API_VERSION), methods=['GET'])
    def get_example():
        num_examples = int(request.args.get('numExamples'))
        return json.dumps({
            'numExamples': num_examples,
            'data': [examples[i] for i in random.sample(range(len(examples)), k=num_examples)]
        })

    @app.route('/api/v{0}/classes'.format(API_VERSION), methods=['GET'])
    def get_classes():
        num_classes = len(classes)
        return json.dumps({
            'numClasses': num_classes,
            'classes': reverse_classes,
        
        })

    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def serve(path):
        if path != '' and os.path.exists(params.dist_dir + path):
            return send_from_directory(params.dist_dir, path)
        else:
            return send_from_directory(params.dist_dir, 'index.html')

    return app


if __name__ == '__main__':
    defaults = util.namespace_json(path=constants.FilePaths.DEFAULTS)
    model_config = config.model_config(defaults).FLAGS
    app = demo(config.gpu_config(), model_config)
    app.run(port=model_config.demo_server_port)

