import os
from setup_logging import setup_logging

setup_logging()
EPOCHS = 1
LEARNING_RATE = 3e-5
LEARNING_RATE_SCHEDULER = 'fixed'
BATCH_SIZE = 50


def set_epochs(epochs):
    global EPOCHS
    EPOCHS = epochs

def set_batch_size(batch_size):
    global BATCH_SIZE
    BATCH_SIZE = batch_size

def set_learning_rate(lr, lr_scheduler):
    global LEARNING_RATE, LEARNING_RATE_SCHEDULER
    LEARNING_RATE = lr
    LEARNING_RATE_SCHEDULER = lr_scheduler
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TOKENIZER = 'BPE'
TOKENIZER_PATH = os.path.join('ListOPs', 'listops_tokenizer')
TOKENIZER_MAX_ENTRIES = 1000000
TOKENIZER_BATCH_SIZE = 2048
TOKENIZER_MAX_LENGTH = 512
VOCAB_SIZE = 100
MODEL_MAX_LENGTH = 512

# SPECIAL_TOKENS = dict(ref_token="#reference#", eq_token="#latex#", sec_token="<sec>", doc_token="<doc>")
LISTOP_OPERATORS = dict(max='MAX',
                        min='MIN',
                        sm='SUM',
                        med='MED',)
KNOWN_TOKENS = [str(i) for i in range(10)]
MAX_TREE_DEPTH = 10
TOKENIZER_VOCAB_SIZE = 0


def set_data_paths(dataset_name):
    global DATA_PATH, DATA_PATH_TRAIN, DATA_PATH_TRAIN_TOKENIZED, DATA_PATH_TEST, DATA_PATH_TEST_TOKENIZED
    DATA_PATH = os.path.join('.', 'data', dataset_name)
    DATA_PATH_TRAIN = os.path.join(DATA_PATH, 'train')
    DATA_PATH_TRAIN_TOKENIZED = os.path.join(DATA_PATH_TRAIN, 'tokenized')
    DATA_PATH_TEST = os.path.join(DATA_PATH, 'test')
    DATA_PATH_TEST_TOKENIZED = os.path.join(DATA_PATH_TEST, 'tokenized')

set_data_paths('listops_MAX_1_1_2_10000_True')

VALIDATION_SPLIT = 5/90
MYSTERIOUS_MAX_ENTRIES = 1000000
N_CONCURRENT_PREPROCESSING_PROCESSES = 8
PREPROCESSING_BATCH_SIZE = 2048
KEEP_IN_MEMORY = False

MAX_SENT_LENGTH = 100
MAX_SEC_LENGTH = 100
MAX_DOC_LENGTH = 100
NUMBER_OF_LABELS = 1
USE_INTERMEDIATE_LOSS = False