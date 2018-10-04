import os
from enum import Enum


class FilePaths(Enum):
    defaults = os.path.abspath('./data/defaults.json')

    def __str__(self):
        return str(self.value)


class FileNames(Enum):
    train_squad_1 = 'train-v1.1.json'
    dev_squad_1 = 'dev-v1.1.json'
    embedding_types = ['word', 'trainable', 'char']
    data_types = ['train', 'val']
    index = '{}_index.json'
    embeddings = '{}_embeddings.txt'
    sent_140 = '{}_sent_140.csv'
    sem_eval = '{}_sem_eval.csv'
