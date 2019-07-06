import os
import csv
import codecs
import yaml
import time
import numpy as np
import torch

from collections import OrderedDict

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def read_config(path):

    return AttrDict(yaml.load(open(path, 'r')))


def read_datas(filename, trans_to_num=False):
    lines = open(filename, 'r').readlines()
    lines = list(map(lambda x: x.split(), lines))
    if trans_to_num:
        lines = [list(map(int, line)) for line in lines]
    return lines


def save_datas(data, filename, trans_to_str=False):
    if trans_to_str:
        data = [list(map(str, line)) for line in data]
    lines = list(map(lambda x: " ".join(x), data))
    with open(filename, 'w') as f:
        f.write("\n".join(lines))


def logging(file):
    def write_log(s):
        print(s, end='')
        with open(file, 'a') as f:
            f.write(s)
    return write_log


def logging_csv(file, header):
    # header: list of names
    i = 1
    fname = file
    while os.path.isfile(fname):
        fname = file + str(i)
        i += 1
    with open(fname, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
    def write_csv(s):
        # s : list of values 
        with open(fname, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(s)
    return write_csv

def logging_dict_csv(file, fieldnames):
    # fieldnames: list of names
    i = 1
    fname = file
    while os.path.isfile(fname):
        fname = file + str(i)
        i += 1
    
    with open(fname, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames,
                                delimiter=',')
        writer.writeheader()
    def write_csv(d, epoch = None, updates = None):
        d = d.copy()
        if epoch:
            d['epoch'] = epoch
        if updates:
            d['updates'] = updates
        with open(fname, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames,
                                    delimiter=',')
            writer.writerow(d)
    return write_csv


def format_time(t):
    return time.strftime("%Y-%m-%d-%H:%M:%S", t)

def combine_results(start_time, end_time, duration, best_epoch, model_type, end_epoch, threshold,  
                yaml_opt, train_loss, val_loss, test_loss):
    # start_time, end_time, duration, model_type, epoch, yaml, train_loss,
    # val_loss, test_loss
    d = OrderedDict()
    d['start_time'] = format_time(start_time)
    d['end_time'] = format_time(end_time)
    d['duration(hr)'] = '{:.2f}'.format(duration / 3600)
    d['best_epoch'] = best_epoch
    d['model_type'] = model_type
    d['end_epoch'] = end_epoch
    d['threshold'] = threshold
    for k,v in yaml_opt.items():
        d[k] = v
    for k,v in train_loss.items():
        d['train_' + k] = v
    for k,v in val_loss.items():
        d['val_' + k] = v
    for k,v in test_loss.items():
        d['test_' + k] = v
    return d
def combine_dict(d1,d2):
    d = {}
    for k, v in d1.items():
        d[k] = v
    for k, v in d2.items():
        d[k] = v
    return d

if __name__ == '__main__':
    ## Debugging
    l = logging_dict_csv('temp.csv',['a','b','c'])
    d = {'a':1,'b':2,'c':3}
    l(d)
    l = logging_csv('temp.csv',['a','b','c'])
    d = [1,2,3]
    l(d)
