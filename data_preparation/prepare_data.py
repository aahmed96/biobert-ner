import sys
#to access helper functions
sys.path.append('..')
#make relevant imports
import numpy as np
import torch
import transformers
import os
import csv
from preprocessing_functions import *

#initialize tokenizer
tokenizer = transformers.BertTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.2', do_lower_case=False)

#load the data
PARENT_DIR = os.path.join('..', 'BioNLP')
CORPUS_PATH = 'BioNLP09-IOB'

#extract sentences and tags from files
train_sentences, train_tags = extractFromDirectories(parent_dir=PARENT_DIR,
                                                     corpus_path=CORPUS_PATH,
                                                     file_type='train.tsv')

val_sentences, val_tags = extractFromDirectories(parent_dir=PARENT_DIR,
                                                     corpus_path=CORPUS_PATH,
                                                     file_type='devel.tsv')

test_sentences, test_tags = extractFromDirectories(parent_dir=PARENT_DIR,
                                                     corpus_path=CORPUS_PATH,
                                                     file_type='test.tsv')

#generate datasets
tokenized_train_dataset = tokenize_dataset(train_sentences,train_tags,tokenizer)
tokenized_val_dataset = tokenize_dataset(val_sentences,val_tags,tokenizer)
tokenized_test_dataset = tokenize_dataset(test_sentences,test_tags,tokenizer)

#save datasets as TensorDataset
save_as_tensor_dataset('prepared_data/train.pt', tokenized_train_dataset)
save_as_tensor_dataset('prepared_data/val.pt', tokenized_val_dataset)
save_as_tensor_dataset('prepared_data/test.pt', tokenized_test_dataset)
