from src import constants, config, util
from train import train
from test import test
from preprocess import preprocess
from demo import demo


def main(sess_config, flags):
    params = flags.FLAGS
    mode = params.mode.lower().strip()

    if mode == constants.Modes.TRAIN:
        train(sess_config, params)
    elif mode == constants.Modes.TEST:
        test(sess_config, params)
    elif mode == constants.Modes.PREPROCESS:
        preprocess(params)
    elif mode == constants.Modes.DEMO:
        demo(sess_config, params)
    else:
        print('Unknown Mode.')
        exit(0)


if __name__ == '__main__':
    defaults = util.namespace_json(path=constants.FilePaths.DEFAULTS)
    main(config.gpu_config(), config.model_config(defaults))
