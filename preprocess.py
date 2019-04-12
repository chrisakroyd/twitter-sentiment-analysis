from src import config, constants, preprocess as prepro, util


def preprocess(params):
    dataset = params.dataset.lower().strip()
    util.make_dirs(util.get_directories(params))

    output_dir = util.processed_data_directory(params)

    # Prompt user for confirmation if this action overwrites existing data.
    if util.directory_exists(output_dir) and not util.directory_is_empty(output_dir):
        if not util.yes_no_prompt(constants.Prompts.DATA_EXISTS):
            exit(0)

    if dataset == constants.Datasets.SEM_EVAL:
        data = prepro.get_data_sem_eval(constants.FilePaths.SEM_EVAL)
        prepro.process(params, data)
    elif dataset == constants.Datasets.SEM_EVAL_2017:
        data = prepro.get_data_sem_eval(constants.FilePaths.SEM_EVAL_2017)
        prepro.process(params, data)
    elif dataset == constants.Datasets.SENT_140:
        data = prepro.get_data_sent_140(constants.FilePaths.SENT_140, max_examples=params.max_examples)
        prepro.process(params, data)
    else:
        raise NotImplementedError('Unsupported dataset: Valid datasets are {}.'.format('squad'))


if __name__ == '__main__':
    defaults = util.namespace_json(path=constants.FilePaths.DEFAULTS)
    preprocess(config.model_config(defaults))
