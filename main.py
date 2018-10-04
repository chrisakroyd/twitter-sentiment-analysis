from src.constants import FilePaths
from src.util import namespace_json
from src.config import model_config
from train import train
from preprocess import preprocess
from demo import demo


def main(flags):
    hparams = flags
    mode = hparams.mode.lower()

    if mode == 'train':
        train(hparams)
    elif mode == 'preprocess':
        preprocess(hparams)
    elif mode == 'demo':
        demo(hparams)
    else:
        print('Unknown Mode.')
        exit(0)


if __name__ == '__main__':
    defaults = namespace_json(path=FilePaths.defaults.value)
    main(model_config(defaults))
