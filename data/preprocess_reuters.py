#!/usr/bin/env python
# -*- coding: utf-8 -*-
from nltk.corpus import reuters
import nltk
import numpy as np
import os,sys
import re

def load_data(valid_percent = 0.1):
    """
    Load the Reuters dataset.

    Returns:
        raw text and raw labels for train, valid, test set.
    """

    nltk.download('reuters') 
    n_classes = 90
    labels = reuters.categories()

    documents = reuters.fileids()
    test = [d for d in documents if d.startswith('test/')]
    train = [d for d in documents if d.startswith('training/')]
    
    docs = {}
    docs['train'] = [reuters.raw(doc_id) for doc_id in train]
    docs['test'] = [reuters.raw(doc_id) for doc_id in test]
    
    ys = {'train': [], 'test': []}
    ys['train'] = [reuters.categories(doc_id) for doc_id in train]
    ys['test'] = [reuters.categories(doc_id) for doc_id in test]
    
    # Validation
    n_valid =int(valid_percent * len(ys['train']))
    np.random.seed(5)
    idxs = np.random.choice(len(ys['train']), n_valid, replace=False)
    idx_set = set(idxs)
    docs['valid'] = []
    ys['valid'] = []
    train_docs = []
    train_y = []
    for idx,(x,y) in enumerate(zip(docs['train'], ys['train'])):
        if idx in idx_set:
            docs['valid'].append(x)
            ys['valid'].append(y)
        else:
            train_docs.append(x)
            train_y.append(y)

    data = {'x_train': train_docs, 'y_train': train_y,
            'x_valid': docs['valid'], 'y_valid': ys['valid'],
            'x_test': docs['test'], 'y_test': ys['test'],
            'labels': labels}
    return data

def dump_data(data, dir_path):
    def write_raw_text(text_list, path):
        with open(path, 'w') as f:
            for text in text_list:
                text = re.sub(' +', ' ', text.lower().replace('\n',' ').replace('.',''))
                f.write(text + '\n')
    def write_raw_label(label_set_list, path):
        with open(path, 'w') as f:
            for label_set in label_set_list:
                for label in label_set:
                    f.write(label + ' ')
                f.write('\n')
    if not os.path.isdir(dir_path): 
        os.makedirs(dir_path)
    write_raw_text(data['x_train'], os.path.join(dir_path,'text_train'))
    write_raw_text(data['x_valid'], os.path.join(dir_path,'text_val'))
    write_raw_text(data['x_test'], os.path.join(dir_path,'text_test'))

    write_raw_label(data['y_train'], os.path.join(dir_path,'label_train'))
    write_raw_label(data['y_valid'], os.path.join(dir_path,'label_val'))
    write_raw_label(data['y_test'], os.path.join(dir_path,'label_test'))


if __name__ == '__main__':
    config = {}
    data = load_data()
    dump_data(data, sys.argv[1])
