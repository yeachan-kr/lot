from lib2to3.pgen2 import token
import os
import sys
import copy
import pickle
import numpy as np
import pandas as pd

# from data.sampler import SubsetSequentialSampler
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# Huggingface imports
from datasets import load_dataset, Dataset, concatenate_datasets, DatasetDict
from transformers import AutoTokenizer

from modeling_bert import BertForSequenceClassification, BertForMultipleChoice

# Global variable
nlp_dataset = None

class SubsetSequentialSampler(torch.utils.data.Sampler):
    r"""Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (int(self.indices[i]) for i in range(len(self.indices)))
    
    def __len__(self):
        return len(self.indices)  

class Logger(object):
    def __init__(self, location):
        self.terminal = sys.stdout
        self.log = open(location, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass
        

def set_random_seed(seed):
    import torch
    import random
    random.seed(seed)
    np.random.seed(seed+1)
    torch.manual_seed(seed+2)
    torch.cuda.manual_seed(seed+3)
    torch.cuda.manual_seed_all(seed+4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_num_classes(dataset: str) -> int:
    """
    Return the number of classes in the given dataset
    Args:
        dataset: Dataset name (e.g., cifar10, cifar100)
    Returns:
        the number of classes
    """
    n_classes = 0
    if dataset == 'sst2' or dataset =='imdb' or dataset == 'mrpc' or dataset == 'qnli':
        n_classes = 2
    if dataset == 'mnli':
        n_classes = 3
    if not n_classes:
        print('No {} dataset in data directory'.format(dataset))
        exit()
    return n_classes


def get_tokenizer(model: str, max_length: int):
    return AutoTokenizer.from_pretrained(model, model_max_length=max_length)

def load_seq2seq_dataset(dataset: str, datadir: str, model: str, max_length: int = 128, num_class=20):
    
    if dataset in ['sst2', 'mrpc', 'qnli', 'mnli', 'stsb']:
        raw_dataset = load_dataset('glue', dataset)
    elif dataset in ['imdb']:
        raw_dataset = load_dataset(dataset)
        # add data index column to raw_dataset
        raw_dataset['train'] = raw_dataset['train'].add_column('idx', np.arange(len(raw_dataset['train'])))

    tokenizer = get_tokenizer(model, max_length=max_length)
    def single_sentence_tokenize_function(example, max_len=128):
        # add samples indexes
        if 'sentence' in example:
            tokenized_inputs = tokenizer(example['sentence'], padding='max_length', truncation=True)
        elif 'text' in example:
            tokenized_inputs = tokenizer(example['text'], padding='max_length', truncation=True)
        tokenized_inputs['index'] = [i for i in range(len(tokenized_inputs['input_ids']))]
        return tokenized_inputs

    def two_sentence_tokenize_function_mrpc(example, max_len=128):
        tokenized_inputs = tokenizer(example['sentence1'], example['sentence2'], padding='max_length', truncation=True)
        return tokenized_inputs

    def two_sentence_tokenize_function_mnli(example, max_len=128):
        tokenized_inputs = tokenizer(example['premise'], example['hypothesis'], padding='max_length', truncation=True)
        return tokenized_inputs

    def two_sentence_tokenize_function_qnli(example, max_len=128):
        tokenized_inputs = tokenizer(example['question'], example['sentence'], padding='max_length', truncation=True)
        return tokenized_inputs

 
    if dataset == 'sst2':
        tokenized_dataset = raw_dataset.map(single_sentence_tokenize_function, batched=True)
        tokenized_dataset = tokenized_dataset.remove_columns(['sentence'])
    elif dataset == 'mrpc' or dataset == 'stsb':
        tokenized_dataset = raw_dataset.map(two_sentence_tokenize_function_mrpc, batched=True)
        tokenized_dataset = tokenized_dataset.remove_columns(['sentence1', 'sentence2'])
    elif dataset == 'mnli':
        tokenized_dataset = raw_dataset.map(two_sentence_tokenize_function_mnli, batched=True)
        tokenized_dataset = tokenized_dataset.remove_columns(['premise', 'hypothesis'])
    elif dataset == 'qnli':
        tokenized_dataset = raw_dataset.map(two_sentence_tokenize_function_qnli, batched=True)
        tokenized_dataset = tokenized_dataset.remove_columns(['question', 'sentence'])
    elif dataset == 'imdb':
        tokenized_dataset = raw_dataset.map(single_sentence_tokenize_function, batched=True)
        tokenized_dataset = tokenized_dataset.remove_columns(['text'])
    tokenized_dataset.set_format("torch")
    return tokenized_dataset

def get_dataloader(dataset, train_bs: int, test_bs: int, dataidxs=None):
    train_dataset = dataset["train"]
    if 'validation_matched' in dataset:
        test_dataset = dataset["validation_matched"]
    elif 'validation' in dataset:
        test_dataset = dataset["validation"]
    elif 'test' in dataset:
        test_dataset = dataset["test"]

    if dataidxs is None:
        train_dl = DataLoader(train_dataset, batch_size=train_bs, pin_memory=True, shuffle=True)
        test_dl = DataLoader(test_dataset, batch_size=test_bs, pin_memory=True, shuffle=False)
    else:
        train_dl = DataLoader(train_dataset, batch_size=train_bs, sampler=SubsetRandomSampler(dataidxs), pin_memory=True)
        test_dl = DataLoader(test_dataset, batch_size=test_bs, pin_memory=True, shuffle=False)
    return train_dl, test_dl

def initialize_networks(alg:str, dataset: str, model: str, device: str ='cpu', n_moe=3):
    """ Initialize the network based on the given dataset and model specification. """
    backbone = BertForSequenceClassification.from_pretrained(model, num_labels=get_num_classes(dataset))
    if alg == 'lot':
        from modeling_bert import BertLoTEncoder
        encoder = BertLoTEncoder(backbone.config)
        encoder.load_state_dict(backbone.bert.encoder.state_dict(), strict=False)
        backbone.bert.encoder = encoder

    return backbone

def load_model(modeldir, filename):
    # load the model from disk
    return pickle.load(open(os.path.join(modeldir, filename), 'rb'))

def save_model(model, modeldir, filename):
    # save the model to disk
    pickle.dump(model, open(os.path.join(modeldir, filename), 'wb'))
