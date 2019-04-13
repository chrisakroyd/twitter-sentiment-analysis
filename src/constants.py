import os


class Datasets:
    """ Constant dataset keys.
        The following keys are defined:
        * SEM_EVAL: Full Sem eval dataset
        * SEM_EVAL_2017: Only sem eval sentiment data released in 2017.
        * SENT_140: Binary Positive/Negative data from sent 140 (http://help.sentiment140.com/for-students)
    """
    SEM_EVAL = 'sem_eval'
    SEM_EVAL_2017 = 'sem_eval_2017'
    SENT_140 = 'sent_140'


class FilePaths:
    """ Constant filepaths for saving/loading data.
        The following keys are defined:
        * DEFAULTS: Path to the default model parameters.
    """
    DEFAULTS = os.path.abspath('./data/defaults.json')
    SEM_EVAL = os.path.abspath('./data/sem_eval/full')
    SEM_EVAL_2017 = os.path.abspath('./data/sem_eval/2017_dataset')
    SENT_140 = os.path.abspath('./data/sent_140/training.1600000.processed.noemoticon.csv')


class FileNames:
    """ Variety of constant filenames.
        The following keys are defined:
        * TRAIN_SQUAD_1: Name of the raw Squad 1 train file.
        * DEV_SQUAD_1: Name of the raw Squad 1 dev file.
        * TRAIN_SQUAD_2: Name of the raw Squad 2 train file.
        * DEV_SQUAD_2: Name of the raw Squad 2 dev file.
        * TRAIN_DEFAULT: Default filename for an unrecognised train dataset.
        * DEV_DEFAULT: Default filename for an unrecognised dev dataset.
        * EXAMPLES: Filename to store examples of data.
        * INDEX: Filename + type for word/character index files.
        * EMBEDDINGS: Filename + type for word/character embedding files.
        * CONTEXT: Filename + type for storing context related information for eval/test files.
        * ANSWERS: Filename + type for storing answer related information for eval/test files.
        * TF_RECORD: Tfrecord template string for storing processed train/dev files.
        * TRAIN: String representing train mode data.
        * DEV: String representing val mode data.
    """
    EXAMPLES = 'examples.json'
    META = 'meta.json'
    CLASSES = 'classes.json'
    INDEX = '{embedding_type}_index.json'
    EMBEDDINGS = '{embedding_type}_embeddings.npy'
    TF_RECORD = '{name}.tfrecord'
    TRAIN = 'train'
    VAL = 'val'


class DirNames:
    """ Constant directory names for storing data/logs/checkpoints.
        The following keys are defined:
        * CHECKPOINTS: Name of the directory to store checkpoint files.
        * LOGS: Name of the directory to store log files.
        * RECORDS: Name of the directory to store .tfrecord files.
        * PROCESSED: Name of the directory to store processed data.
        * EMBEDDINGS: Name of the directory to store raw embeddings.
        * SQUAD_1: Name of the squad v1 directory.
        * SQUAD_2: Name of the squad v2 directory.
    """
    CHECKPOINTS = 'checkpoints'
    LOGS = 'logs'
    RECORDS = 'records'
    PROCESSED = 'processed'
    EMBEDDINGS = 'embeddings'
    SQUAD_1 = Datasets.SEM_EVAL
    SQUAD_2 = Datasets.SENT_140


class Modes:
    """ Standard names for repo modes.
        The following keys are defined:
        * TRAIN: training mode.
        * TEST: testing mode.
        * PREPROCESS: preprocess mode.
        * DEBUG: Debug mode.
        * DEMO: inference mode.
        * DOWNLOAD: download mode.
    """
    DEBUG = 'debug'
    DEMO = 'demo'
    PREPROCESS = 'preprocess'
    TEST = 'test'
    TRAIN = 'train'


class EmbeddingTypes:
    """ Names for embedding types.
        The following keys are defined:
        * WORD: Word embeddings name.
        * TRAINABLE: Trainable embeddings name.
        * CHAR: Character embeddings name.
    """
    WORD = 'word'
    TRAINABLE = 'trainable'
    CHAR = 'char'
    POS = 'pos'

    @staticmethod
    def as_list():
        """ Returns a list of all supported embedding types """
        return [EmbeddingTypes.WORD, EmbeddingTypes.TRAINABLE, EmbeddingTypes.CHAR, EmbeddingTypes.POS]


class ErrorMessages:
    """ Constant error messages.
        The following keys are defined:
        * NO_TEXT: Key for text missing.
        * NO_TOKENS: Key for tokens missing.
        * INVALID_TEXT: Text field is invalid.
        * INVALID_TOKENS: Tokens field is invalid.
        * INVALID_WARMUP_STEPS: warmup_steps parameter is negative.
    """
    NO_TEXT = 'Text key missing from body of POST request.'
    NO_TOKENS = 'Text key missing from body of POST request.'
    INVALID_TEXT = 'Text must be longer than 0 excluding space characters.'
    INVALID_TOKENS = 'Text must be longer than 0 excluding space characters.'
    INVALID_WARMUP_STEPS = 'Warmup steps parameter cannot be negative, got {steps}.'


class Prompts:
    """ Prompt messages for asking user actions
        DATA_EXISTS: Prompt for confirming a non-reversible overwriting of data.
    """
    DATA_EXISTS = 'Preprocessed data already exists for this dataset, would you like to overwrite?'
    FOUND_CONFIG_NO_CHECKPOINTS = 'Found config file at {path} without any checkpoints, would you like to use this config?'
